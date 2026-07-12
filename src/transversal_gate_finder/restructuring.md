# Restructuring
Let's make some rather big refactoring changes to make things more structured.

## Helper classes
High-level idea:
- 2-groups defined coefficients of phase functions (corresponding to diagonal ansatz gates). Since there's infinitely many possible ansatz gates/phase functions, and lin.Hom 2-group homomorphisms are in dense format, we need to store the maximum support. This is PhaseLocs or TIPhaseLocs (in the translation-invariant case).
- The target space for a TI hom is infinite, so we also need to specify a support of coordinates and internal generators. This is the attribute ti_support of TITwoGroupHom
- Note the difference between TIZ2Hom and TITwoGroupHom: TIZ2Hom is sparse in the sense that every column only stores qubit specifiers in the support of that columns (all coefficients are 1 in the Z2 case). TITwoGroupHom is dense (attribute h), but we do need to store the common support of *all* columns together in ti_support otherwise there would be an infinite amount of data. Part of the point of the classes is to shield such implementation differences.

Introduce the following classes:

- Z2Hom:
    - Main attribute: List of list representing a Z2 homomorphism (i.e. sparse representation of coeff matrix)
    - dim0 and dim1 as additional integer attributes.
    - Use for: GateFinder.check, .logicals, and .other_checks
    - Method: transpose
    - Method: composition with @ (implement either manually in sparse form or by conversion to numpy array). use for example in "test_commutation".
    - Method: phase_pullback: Takes as non-self argument a PhaseLocs object. Returns TwoGroupHom with .phase_locs0 equal to input PhaseLocs, and .phase_locs0 equal to what was previously returned as second return by "pullback_homomorphism". Replaces "pullback_homomorphism" function

- TIZ2Hom:
    - Wraps main attribute: h: List of list of "TI qubit specifiers", pairs of (coord-tuple, internal-qubit-nr)
    - Additional attributes: dim0, dim1: dimensions "per unit cell"
    - Use for: TIGateFinder.checks, .other_checks
    - Methods: transpose, composition with @
    - Method: phase_pullback: Takes TIPhaseLoc as non-self argument. Returns TITwoGroupHom with two TIPhaseLoc as .phase_loc0 and .phase_loc1. This method is sth in between "ti_pullback_homomorphism" and "ti_local_pullback", see second section below

- PhaseLocs:
    - Wraps attribute: locs: list, locs[l] is of length dim[l] each contains one "gate location" (a set of qubit numbers)
    - Also keeps total qubit number as attribute
    - Used for (and same format as): GateFinder.gates (with TI phase_locs attribute)

- TIPhaseLocs:
    - Wraps attribute: locs: list, locs[l] is a list of *normalized* "TI gate location" sets of (coord, internal-qubit-nr) pairs. len(phase_locs[l]) is equal to the number of internal-gate-nrs of level l.
    - Also keeps qubit number per unit cell as attribute
    - (In general, "TI gate locations" should always be normalized internally in the code, and only be in unnormalized form as user input and as output (see TITwoGroupHom.to_string). you can add an "add_locs" method here that does normalization automatically.)

- TwoGroupHom:
    - Main attribute: h: lin.Hom object
    - Optional attributes: phase_locs0, phase_locs1: None, PhaseLocs or TIPhaseLocs objects
    - Method: composition A @ B if B.phase_locs1 = A.phase_locs0
    - If needed: method transpose
    - Used for things like GateFinder.stabphys_allphys (with phase_locs0=None) or GateFinder.transphys_allphys

- TITwoGroupHom:
    - attribute: h: lin.Hom object
    - attribute: ti_support: List such that ti_support[l] is a list of (coordinate, internal-gate-nr) pairs for 2-group level l. the internal-gate-nrs start from 0 for each level l. len(ti_support[l] = h.dim1[l]). This stores the target support of a map between 2-group Z free modules, whereas h stores the coefficients.
    - Optional attribute phase_locs0 (None if not set): PhaseLocs or TIPhaseLocs object describing phase locations for source 2-group
    - Optional attribute phase_locs1: PhaseLocs or TIPhaseLocs object describing phase locations for target 2-group
    - Method: composition A @ B if B.phase_locs1 == A.phase_locs0
    - Method: ti_sum: sum over all coordinates. Results in a TwoGroupHom. phase_locs0 and phase_locs1 stay unchanged (same object)
    - to_string: prints non-zero coefficients of each column in the lin.Hom attribute. With each coefficient, it prints either (1) just the (coord, internal-gate-nr) pair if dim1.phase_locs is None, or (1) phase_locs[l][internal-gate-nr] with all coords shifted by the coord of the (coord, internal-gate-nr) pair. this replaces the function "gates_string". 
    - Used for TIGateFinder.local_gates, output of TIZ2Hom.phase_pullback

Use these classes as much as possible. For every method of GateFinder or TIGateFinder, check whether they make sense as methods of TwoGroupHom, TITwoGroupHom, etc. I have used some terminology referring to "qubits", "gates", etc above so you and I understand the context for how these things are used better, but the goal is to keep naming neutral mathematical in terms of "twogroups", "phase functions", "phase function support", etc, for all of the above classes and their methods. (Ofc, for GateFinder and TIGateFinder, we do still want to use "qubit/QEC" terminology). Please add type hints everywhere to the code where it seems helpful for me to understand your refactoring.

## restructure pullback functions
- The methods Z2Hom.phase_pullback and TIZ2Hom.phase_pullback are the "standard pullback".
- Z2Hom.phase_pullback is the same as pullback_homomorphism as implemented, but with TwoGroup and TwoGroupHom inputs and outputs.
- TIZ2Hom.phase_pullback is similar to the "local" pullback but without the special "local_gates" parameter. It should map each TI ansatz gate (normalized set of qubit specifiers) on the qubits to a phase function on the checks which include both coordinate and internal number. Like in the current "ti_pullback_homomorphism", you should normalize all target (check) gate locations, but you should keep the shift coordinate for the ti_support attribute of the TITwoGroupHom output.
- You should be able to reuse a lot of the code for the non-TI and TI .phase_pullback methods, like you already do in the current implementation.
- The current "ti_pullback_homomorphism" replaced by the ti_sum of the new standard pullback TIZ2Hom.phase_pullback
- We're gonna change the implementation of the TIGateFinder.local_gates attribute: Instead of just being a subset of the ansatz gates (shifted to prescribed coordinates), we're now gonna implement it as a Hom from some finite 2-group to the ansatz gates (and their translates). So this is a TITwoGroupHom object without phase_locs0 attribute but phase_locs1 attribute pointing to TIGateFinder.gates. Thus the current "ti_local_pullback" is now just the composition TIZ2Hom.phase_pullback(...) @ TIGateFinder.local_gates



