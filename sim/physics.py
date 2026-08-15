"""
Mechanistic decoder: composition + architecture + time -> real property trajectories.

This module contains ZERO learned parameters. It is the fixed physics half of the
framework. The ML half predicts only the handful of parameters marked PREDICTED below;
everything here is textbook and can be defended line by line in a viva.

Chain of reasoning:
    weight fractions  --rho-->  volume fractions
    Halpin-Tsai       ------->  Es, solid-phase composite modulus
    autocatalytic ODE ------->  Mn(t), molecular weight decay
    solubility cutoff ------->  mass(t), and hence porosity(t)
    Gibson-Ashby      ------->  E(t) = C * Es(t) * (1 - P(t))^n

PARAMETER PROVENANCE - read before trusting any number:
  * Polymer/ceramic moduli and densities are handbook values, reliable to ~+-20%.
  * Hydrolysis rate constants k are ORDER-OF-MAGNITUDE PRIORS chosen to reproduce
    commonly reported degradation windows. They are starting values to be REFIT from
    the Tier 3 curated dataset, not ground truth. Treat them as a prior mean, and say
    so in the thesis. As calibrated, time to 50% mass loss at P=0.6 is:
        PLGA 50:50  1.4 mo   (reported 1-2 mo)
        PLGA 85:15  3.2 mo   (reported 3-6 mo)
        PLLA        1.4 yr   (reported 1-3 yr)
        PCL         2.6 yr   (reported 2-3 yr)
    tools/selftest.py holds the ordering; these magnitudes are the calibration claim.
  * n and C are PREDICTED per architecture - fit them from the simulated tier
    (data/gibson_ashby_fits.csv, fitted from data/simulated_all.csv) rather than assuming.
"""

import numpy as np

# --------------------------------------------------------------------------
# Material property tables
# --------------------------------------------------------------------------

POLYMERS = {
    #            E_MPa  rho    Mn0_kDa  k_hyd(1/wk)  Mc_kDa  crystallinity
    # k re-solved against the 2-3 yr window after the rate law was made saturating
    # (see acid_activity); 0.004 belonged to the old unbounded form and now reads 4.6 yr.
    "PCL":        dict(E=350,  rho=1.145, Mn0=80,  k=0.007, Mc=10, Xc=0.45),
    "PLLA":       dict(E=3500, rho=1.240, Mn0=100, k=0.012, Mc=15, Xc=0.40),
    "PLGA_85_15": dict(E=2200, rho=1.300, Mn0=90,  k=0.055, Mc=15, Xc=0.10),
    "PLGA_50_50": dict(E=2000, rho=1.340, Mn0=60,  k=0.130, Mc=15, Xc=0.00),
    "chitosan":   dict(E=1500, rho=1.340, Mn0=200, k=0.020, Mc=30, Xc=0.30),
    "gelatin":    dict(E=500,  rho=1.350, Mn0=100, k=0.300, Mc=20, Xc=0.00),
    "silk":       dict(E=2500, rho=1.350, Mn0=150, k=0.010, Mc=25, Xc=0.50),
    "PHB":        dict(E=3500, rho=1.250, Mn0=300, k=0.008, Mc=20, Xc=0.60),
}

# Buffering scales with how readily the ceramic DISSOLVES, not merely with how much
# calcium it contains. Stoichiometric HA is nearly inert in vivo and buffers weakly;
# beta-TCP resorbs; 45S5 bioglass releases alkaline ions rapidly. Getting this ordering
# right is what reproduces the reported "ceramic slows polyester degradation" effect.
CERAMICS = {
    #                 E_MPa    rho    solubility-weighted buffering capacity
    "none":        dict(E=0.0,    rho=1.0,  buffer=0.0),
    "HA":          dict(E=100000, rho=3.16, buffer=0.3),
    "beta_TCP":    dict(E=60000,  rho=3.07, buffer=1.0),
    "bioglass":    dict(E=35000,  rho=2.70, buffer=1.8),
    "CaCO3":       dict(E=45000,  rho=2.71, buffer=1.5),
}

# Native bone reference envelopes (compressive modulus, MPa)
BONE_SITES = {
    "trabecular_low":  dict(E_native=50,   sigma_req=2.0),
    "trabecular_mid":  dict(E_native=300,  sigma_req=6.0),
    "trabecular_high": dict(E_native=800,  sigma_req=12.0),
    "cortical":        dict(E_native=15000, sigma_req=130.0),
}


# --------------------------------------------------------------------------
# Composition -> solid-phase modulus
# --------------------------------------------------------------------------

def weight_to_volume_fraction(w_ceramic, rho_poly, rho_ceramic):
    """Convert ceramic weight fraction to volume fraction. Papers report wt%; models need vol%."""
    # Guarded rather than assumed: a FRACTION is expected and a percentage is the obvious
    # slip, since every source paper prints wt%. Unguarded, w = 30 leaves the polymer with
    # negative volume and the whole chain downstream returns a confident negative modulus
    # instead of failing. Same class of error tools/validate_curation.py gates Tier 3 for.
    if not 0.0 <= w_ceramic < 1.0:
        raise ValueError(
            f"ceramic weight fraction must be in [0, 1), got {w_ceramic}"
            + (" - looks like a percentage; divide by 100" if w_ceramic >= 1.0 else ""))
    if w_ceramic <= 0:
        return 0.0
    vc = w_ceramic / rho_ceramic
    vp = (1.0 - w_ceramic) / rho_poly
    return vc / (vc + vp)


def halpin_tsai(Em, Ef, Vf, xi=2.0):
    """Halpin-Tsai composite modulus. xi ~ 2 for roughly spherical particulate filler."""
    if Vf <= 0 or Ef <= 0:
        return Em
    r = Ef / Em
    eta = (r - 1.0) / (r + xi)
    return Em * (1.0 + xi * eta * Vf) / (1.0 - eta * Vf)


def solid_modulus(polymer, ceramic="none", ceramic_wt=0.0, xi=2.0):
    """Solid-phase (fully dense) composite modulus in MPa."""
    p, c = POLYMERS[polymer], CERAMICS[ceramic]
    Vf = weight_to_volume_fraction(ceramic_wt, p["rho"], c["rho"])
    return halpin_tsai(p["E"], c["E"], Vf, xi), Vf


# --------------------------------------------------------------------------
# Degradation kinetics
# --------------------------------------------------------------------------

def acid_activity(Mn_rel):
    """
    Scission-generated acid, as a bounded activity in [0, 1).

    Chain-end concentration goes as 1/Mn, which diverges as the chains shorten. A rate
    law linear in that quantity has no upper bound and drives Mn to zero in a single
    step; the physical amplification saturates instead, because once the medium is
    acidic more acid changes little. The SAME activity drives both the rate law and the
    pH mapping - they are two readings of one acid concentration, and letting them
    disagree is how buffering becomes visible in pH but not in the kinetics.
    """
    raw = np.maximum(0.0, 1.0 / np.maximum(Mn_rel, 1e-12) - 1.0)
    return raw / (1.0 + raw)


def degrade(polymer, ceramic="none", ceramic_wt=0.0, porosity=0.6,
            weeks=np.arange(0, 53, 1.0), alpha=6.0, gamma=0.6, p_sol=4.0):
    """
    Autocatalytic hydrolysis of the polymer phase.

        dm/dt = -k * m * (1 + alpha * acid * (1 - gamma*P) * (1 - buffer))

    where m = Mn/Mn0 and `acid` is the scission-generated carboxylic acid, which is
    (a) amplified by its own concentration (autocatalysis), (b) washed out faster in a
    more porous scaffold, and (c) neutralised by dissolving ceramic. This is the
    mechanism behind the well-known result that HA/TCP content SLOWS polyester
    degradation - a good SHAP sanity check for the ML half.

    Returns dict of arrays over `weeks`.
    """
    p, c = POLYMERS[polymer], CERAMICS[ceramic]
    _, Vf = solid_modulus(polymer, ceramic, ceramic_wt)
    buffer = min(0.90, c["buffer"] * Vf * 4.0)

    # crystalline domains resist hydrolysis
    k0 = p["k"] * (1.0 - 0.5 * p["Xc"])
    washout = max(0.0, 1.0 - gamma * porosity)

    m = 1.0
    Mn, dt = [], np.diff(weeks, prepend=weeks[0])
    for step in dt:
        k_eff = k0 * (1.0 + alpha * acid_activity(m) * washout * (1.0 - buffer))
        # Exponential update, not m - k*m*dt: the explicit form overshoots into negative
        # m once k_eff*dt > 1 (which autocatalysis reaches within weeks for PLGA 50:50),
        # and the clamp that used to catch it pinned every formulation to the same floor -
        # erasing exactly the buffering difference this model exists to show.
        m *= np.exp(-k_eff * step)
        Mn.append(m)
    Mn = np.array(Mn)

    Mn_kDa = Mn * p["Mn0"]
    # Chains dissolve once they fall below the critical entanglement/solubility MW. The
    # sigmoid is not quite 0 at full molecular weight, so it is referenced to its own t=0
    # value: a scaffold that has not degraded has lost no mass, by definition.
    f_soluble = 1.0 / (1.0 + (Mn_kDa / p["Mc"]) ** p_sol)
    polymer_mass = np.clip((1.0 - f_soluble) / (1.0 - f_soluble[0]), 0.0, 1.0)

    # ceramic is retained far longer than the polymer; total mass is the weighted sum
    mass = (1.0 - ceramic_wt) * polymer_mass + ceramic_wt * np.ones_like(polymer_mass)

    # Same saturating acid activity that drives the rate law above. Porosity lets acid
    # diffuse out, so it raises local pH.
    pH = 7.4 - 3.0 * acid_activity(Mn) * (1.0 - buffer) * (1.0 - 0.7 * porosity)

    return dict(weeks=weeks, Mn_rel=Mn, Mn_kDa=Mn_kDa,
                mass_remaining=mass, polymer_mass=polymer_mass,
                pH=pH, buffer=buffer)


# --------------------------------------------------------------------------
# The coupled trajectory - this is the centre of the framework
# --------------------------------------------------------------------------

def property_trajectory(polymer, ceramic="none", ceramic_wt=0.0, porosity0=0.6,
                        n=2.0, C=1.0, a=1.0, weeks=np.arange(0, 53, 1.0)):
    """
    Full coupled prediction.

    n, C : Gibson-Ashby topology parameters   <- PREDICTED by the ML half
    a    : modulus/molecular-weight coupling  <- PREDICTED by the ML half

    The loop that matters:
        degradation -> mass loss -> porosity UP and solid modulus DOWN -> E(t) falls
        faster than either effect alone would suggest.
    """
    Es0, _ = solid_modulus(polymer, ceramic, ceramic_wt)
    deg = degrade(polymer, ceramic, ceramic_wt, porosity0, weeks)

    P_t = 1.0 - (1.0 - porosity0) * deg["mass_remaining"]
    Es_t = Es0 * deg["Mn_rel"] ** a
    E_t = C * Es_t * np.clip(1.0 - P_t, 1e-6, 1.0) ** n

    # Gibson-Ashby strength scaling; sigma_s taken as a fixed fraction of Es
    sigma_s = 0.035 * Es_t
    sigma_t = 0.3 * sigma_s * np.clip(1.0 - P_t, 1e-6, 1.0) ** 1.5

    return dict(weeks=weeks, E=E_t, sigma=sigma_t, porosity=P_t,
                mass_remaining=deg["mass_remaining"],
                polymer_mass=deg["polymer_mass"], Mn_kDa=deg["Mn_kDa"],
                pH=deg["pH"], E0=float(E_t[0]), Es0=Es0,
                ceramic_retained=ceramic_wt)


# --------------------------------------------------------------------------
# Bone healing and the load-transfer objective
# --------------------------------------------------------------------------

def bone_ingrowth(weeks, site="trabecular_mid", pore_um=350.0, porosity=0.6,
                  t_half=12.0, tau=3.5):
    """
    Sigmoidal new-bone modulus, modulated by how osteoconductive the architecture is.
    Pore size has a well-established optimum near 300-400 um: too small blocks
    vascularisation, too large cuts the surface available for cell attachment.
    """
    E_native = BONE_SITES[site]["E_native"]
    pore_quality = np.exp(-0.5 * ((pore_um - 350.0) / 180.0) ** 2)
    porosity_quality = np.clip((porosity - 0.35) / 0.30, 0.0, 1.0)
    capacity = 0.35 + 0.65 * pore_quality * porosity_quality
    return E_native * capacity / (1.0 + np.exp(-(weeks - t_half) / tau))


def load_transfer_cost(traj, site="trabecular_mid", pore_um=350.0):
    """
    Mismatch between total construct stiffness and native bone, integrated over healing.
    Lower is better. This is the objective the inverse-design stage minimises.
    """
    E_native = BONE_SITES[site]["E_native"]
    E_bone = bone_ingrowth(traj["weeks"], site, pore_um, traj["porosity"][0])
    total = traj["E"] + E_bone
    mismatch = np.abs(total - E_native) / E_native
    return float(np.trapezoid(mismatch, traj["weeks"]) / (traj["weeks"][-1] - traj["weeks"][0])), \
        total, E_bone


def design_report(polymer, ceramic, ceramic_wt, porosity0, n, C,
                  site="trabecular_mid", pore_um=350.0, a=1.0):
    """Evaluate one candidate scaffold against every clinical constraint."""
    traj = property_trajectory(polymer, ceramic, ceramic_wt, porosity0, n, C, a)
    cost, total, E_bone = load_transfer_cost(traj, site, pore_um)
    req = BONE_SITES[site]
    # Judge resorption on the POLYMER phase: the ceramic is retained on this timescale,
    # so a total-mass criterion is unreachable for any ceramic-loaded scaffold.
    resorbed = traj["weeks"][traj["polymer_mass"] < 0.10]
    return dict(
        E0_MPa=round(traj["E0"], 1),
        sigma0_MPa=round(float(traj["sigma"][0]), 2),
        stiffness_at_12wk_pct=round(100 * float(traj["E"][12]) / traj["E0"], 1),
        mass_at_12wk_pct=round(100 * float(traj["mass_remaining"][12]), 1),
        polymer_resorbed_week=float(resorbed[0]) if len(resorbed) else np.nan,
        min_pH=round(float(traj["pH"].min()), 2),
        load_transfer_cost=round(cost, 4),
        meets_stiffness=bool(0.5 * req["E_native"] <= traj["E0"] <= 2.0 * req["E_native"]),
        meets_strength=bool(traj["sigma"][0] >= req["sigma_req"]),
        pH_safe=bool(traj["pH"].min() >= 6.5),
    ), traj
