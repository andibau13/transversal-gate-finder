additional small API changes/simplification/cleanup in core.py
- move matrix_from_list and list_from_matrix into Z2Hom.to_array and from_array.
- in TwoGroupHom.__init__, test if format of phase_locs0/1 matches h.dim0/1
- introduce TwoGroupElem and TITwoGroupElem classes with wrapping a lin.Hom.Elem, with an optional gate_loc attribute. And TITwoGroupElem has an additional ti_support attribute.
- TwoGroupHom, TITwoGroupHom, TwoGroupElem, TITwoGroupElem can only be composed if they share the literal same phase_locs-type objects (references to the same thing which is think is standard in python). make sure that you never make a "deep copy" of a phase_loc-type object. So you can replace the manual equality check with a simple "is" (or is this what == means in python?)
- remove all the one-liner wrapper in GateFinder between add_checks to add_gates (".checks.add_colums" is as good as ".add_checks" etc)
- move all methods of GateFinder adding gate configurations (such as all_single_qubit, .._in_groups, etc) to the GateLocs class
- replace "remove_redundant_checks" and "remove_redundant_logicals" methods by a Z2Hom.remove_redundant_columns method
- remove the one-line wrappers "pullback_checks" and "pullback_logicals".

- make the analogous changes in translation_invariance.py
- In general, offload functions to the helper classes if possible