# RateLab_Project


This repo is a **learning-by-coding** implementation of a *small* thermonuclear reaction network (an **α-capture chain**) using **REAClib** rates and a simple **one‑zone** (parameterized) thermodynamic trajectory. The goal is to connect the *physics definitions and equations* (Iliadis) to the actual numbers your code evolves.

> Primary reference: Christian Iliadis, *Nuclear Physics of Stars* (2nd ed., 2015).

---

## 1) Core physical quantities (what we evolve)

### Number density, mass density
- **Number density** of species *i*:  
  \[
  N_i \equiv \text{nuclei of species } i \text{ per unit volume}\quad [\mathrm{cm^{-3}}]
  \]
- **Mass density**:
  \[
  \rho = \frac{1}{N_A}\sum_i N_i M_i
  \]
  where \(M_i\) is the **relative atomic mass** (in u) and \(N_A\) is Avogadro’s constant. fileciteturn15file0L1-L5

### Mass fraction \(X_i\) and mole fraction / molar abundance \(Y_i\)
Iliadis defines:
\[
X_i \equiv \frac{N_i M_i}{\rho N_A}
\qquad\text{and}\qquad
Y_i \equiv \frac{X_i}{M_i} = \frac{N_i}{\rho N_A}
\]
- \(X_i\): fraction of *mass* in species \(i\) (dimensionless) fileciteturn15file0L23-L33  
- \(Y_i\): **mole fraction / molar abundance** (often treated as “mol per gram” in practice); it stays constant under pure expansion/compression if no reactions occur (useful numerically). fileciteturn15file0L34-L38  

**Useful identity** (from \(Y_i = N_i/(\rho N_A)\)):
\[
N_i = \rho N_A Y_i
\]
This is what turns *per‑volume* rates into *ODEs for \(Y\)*. fileciteturn15file0L29-L33

---

## 2) Thermonuclear reaction rates \(N_A\langle\sigma v\rangle\)

### From “pairs collide” to \(\langle\sigma v\rangle\)
For a two‑body reaction \(0+1 \to ...\), Iliadis starts with:
\[
r_{01} = N_0 N_1 \int_0^\infty v\,P(v)\,\sigma(v)\,dv \equiv N_0 N_1\langle\sigma v\rangle_{01}
\]
(where \(r_{01}\) is reactions per unit volume per unit time). fileciteturn14file1L18-L24

For **identical reactants**, the number of distinct pairs is reduced by 1/2; Iliadis writes the general expression using a Kronecker‑delta factor. fileciteturn14file1L25-L35  
(In our α‑captures, reactants are different, so this factor is 1.)

### Maxwell–Boltzmann distribution \(P(v)\) and energy form
At thermodynamic equilibrium (non‑degenerate, non‑relativistic), relative velocities are Maxwellian:
\[
P(v)\,dv = \left(\frac{m_{01}}{2\pi kT}\right)^{3/2} e^{-m_{01}v^2/(2kT)} 4\pi v^2\,dv
\]
and can be rewritten as an **energy distribution** \(P(E)\,dE\). fileciteturn14file0L1-L27

With that, Iliadis obtains the standard Maxwellian average:
\[
\langle\sigma v\rangle_{01}
= \left(\frac{8}{\pi m_{01}}\right)^{1/2}\frac{1}{(kT)^{3/2}}
\int_0^\infty E\,\sigma(E)\,e^{-E/kT}\,dE
\]
fileciteturn14file0L29-L44

### Why everyone tabulates \(N_A\langle\sigma v\rangle\) and what \(T_9\) is
In practice, the literature typically tabulates:
- \(N_A\langle\sigma v\rangle\) in \([\mathrm{cm^3\,mol^{-1}\,s^{-1}}]\) fileciteturn14file1L37-L39

Iliadis also gives a convenient numerical form:
\[
N_A\langle\sigma v\rangle
= 3.7318\times 10^{10}\,
\frac{1}{T_9^{3/2}}
\sqrt{\frac{M_0+M_1}{M_0 M_1}}
\int_0^\infty E\sigma(E)\,e^{-11.605\,E/T_9}\,dE
\]
(with \(E\) in MeV, \(T_9 \equiv T/10^9\ \mathrm{K}\), \(\sigma\) in barns). fileciteturn14file0L45-L60

And explicitly:
\[
T_9 \equiv \frac{T}{10^9\ \mathrm{K}}
\]
fileciteturn14file0L57-L59

**Interpretation:** \(N_A\langle\sigma v\rangle(T_9)\) is the *thermally averaged microscopic probability* of reaction, folded with the Maxwellian velocities.

---

## 3) Turning rates into abundance ODEs

Using \(N_i=\rho N_A Y_i\), a **two‑body capture** \(i+j\to k\) has (schematically):
\[
\frac{dY_i}{dt}\sim -\rho\,Y_iY_j\,N_A\langle\sigma v\rangle_{ij\to k},
\qquad
\frac{dY_k}{dt}\sim +\rho\,Y_iY_j\,N_A\langle\sigma v\rangle_{ij\to k}.
\]
This is exactly the structure implemented in `ratelab/network.py` for each α‑capture step. fileciteturn15file6L30-L45

A **photodisintegration** (reverse) reaction acts like a **decay** with rate (decay constant) \(\lambda_\gamma\):
\[
\frac{dY_{\text{parent}}}{dt} = -\lambda_\gamma\,Y_{\text{parent}}
\]
(see Iliadis’ general decay‑constant definition). fileciteturn15file2L40-L45

---

## 4) Forward ↔ reverse (photodisintegration) and detailed balance

For capture ↔ photodisintegration pairs, Iliadis shows that the **reverse photodisintegration decay constant** can be computed from the forward stellar rate using spin factors, partition functions, masses, and an exponential Boltzmann factor in \(Q/T_9\). Example 3.2 explicitly evaluates:
\[
\lambda_\gamma^\*(\text{reverse})
\propto T_9^{3/2}\,e^{-11.605\,Q/T_9}\,N_A\langle\sigma v\rangle^\*_{\text{forward}}
\]
(with additional multiplicative factors from spins and partition functions). fileciteturn14file4L18-L47

### How we do it in this repo (toy approach)
For learning and rapid prototyping, we allow **two options**:

1) **Use REAClib-provided reverse fits**: if the library already provides the reverse reaction as its own fit, we can evaluate \(\lambda_\gamma(T_9)\) directly with the same 7‑parameter form (this is what your current network code structure assumes).

2) **Derive \(\lambda_\gamma\) from the forward rate** (more “physics transparent”): implement the detailed‑balance formula (as in Iliadis Example 3.2) using \(Q\)-values, spins, and partition functions.

Option (1) is fast and consistent with how large networks are often wired; option (2) is excellent for *mastery* because you see every physical factor.

---

## 5) REAClib 7‑parameter fit (what our code evaluates)

REAClib rates are provided as **fits** to tabulated \(N_A\langle\sigma v\rangle(T_9)\), designed for very fast evaluation. For one coefficient set \(a_1,\dots,a_7\),
\[
N_A\langle\sigma v\rangle(T_9)
= \exp\!\Big(
a_1 + a_2T_9^{-1} + a_3T_9^{-1/3}
+ a_4T_9^{1/3} + a_5T_9 + a_6T_9^{5/3} + a_7\ln T_9
\Big)
\]
(as implemented in `ratelab/reaclib.py`). fileciteturn14file8L3-L26

Many reactions have **multiple sets** (different temperature regions / components). We sum them:
\[
\text{rate}(T_9) = \sum_k \exp(E_k(T_9))
\]
(as implemented in `ratelab/rates.py`). fileciteturn14file5L48-L68

### Temperature sensitivity
We also compute:
\[
\frac{d\ln(\text{rate})}{d\ln T_9}
\]
both analytically (chain rule) and by finite‑difference for sanity checks. fileciteturn14file10L22-L51

---

## 6) The toy network we evolve

We start with a minimal α‑chain relevant to O/Si‑region processing:

- \(^{16}\mathrm{O}(\alpha,\gamma)^{20}\mathrm{Ne}\)
- \(^{20}\mathrm{Ne}(\alpha,\gamma)^{24}\mathrm{Mg}\)
- \(^{24}\mathrm{Mg}(\alpha,\gamma)^{28}\mathrm{Si}\)
- \(^{28}\mathrm{Si}(\alpha,\gamma)^{32}\mathrm{S}\)

Forward ODE structure is in `ratelab/network.py`. fileciteturn15file6L22-L45

---

## 7) One‑zone “shock / explosive burning” trajectory

A common approach in explosive nucleosynthesis is to prescribe \(T(t)\) and \(\rho(t)\) with a simple parameterization once \(T_{\text{peak}}\) and \(\rho_{\text{peak}}\) are chosen. Iliadis discusses this idea and gives an exponential density falloff with an expansion timescale \(\tau_{\mathrm{exp}}\), e.g.:
\[
\rho(t) = \rho_{\mathrm{peak}}\,e^{-t/\tau_{\mathrm{exp}}}
\]
fileciteturn19file2L1-L7

In this repo we use a simplified “shock_trajectory” parameterization in code (see `ratelab/trajectory.py`), and wire it into the integrator in `scripts/run_onezone.py`. fileciteturn15file5L13-L28

---

## 8) Fluxes (what is “driving” the network?)

A helpful diagnostic is the **abundance flux** for each reaction, i.e. the instantaneous flow of material through a link.

For an α‑capture \( \alpha + i \to k\):
\[
F_{\alpha i\to k}(t) = \rho(t)\,Y_\alpha(t)\,Y_i(t)\,N_A\langle\sigma v\rangle_{\alpha i\to k}(T_9(t))
\]

For a photodisintegration \(k\to \alpha + i\):
\[
F_{k\to \alpha i}(t) = \lambda_\gamma(T_9(t))\,Y_k(t)
\]

(Your `fluxes()` helper in `network.py` should return these as a dictionary so `run_onezone.py` can plot them.)

---

## 9) Running the demo

### Plot rate curves + sensitivities
```bash
python scripts/plot_rates.py
