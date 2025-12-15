import pynucastro as na
import numpy as np
import re

def extract_sets(original_source: str):
    """
    Parse 7-coefficient sets from a REACLIB 'original_source' text block.
    Assumes the classic formatting: header line then 2 coefficient lines.
    """

    lines = [l.rstrip("\n") for l in original_source.splitlines() if l.strip()]
    sets = []

    i = 0
    while i < len(lines):
        if re.fullmatch(r"\d+", lines[i].strip()):
            # coefficients from DB
            coeff1 = lines[i+2]
            coeff2 = lines[i+3]
            nums = re.findall(r"[+-]?\d+\.\d+e[+-]?\d+", coeff1 + " " + coeff2)

            if len(nums) != 7:
                raise ValueError(f"Expected 7 coefficients, got {len(nums)} at block starting line {i}")
            
            sets.append([float(x) for x in nums])
            i += 4
        else:
            i += 1

    return np.array(sets, dtype=float)

rl = na.ReacLibLibrary()

for name in ["o16(a,g)ne20", "ne20(a,g)mg24", "mg24(a,g)si28", "si28(a,g)s32"]:
    r = rl.get_rate_by_name(name)
    print("\n===", name, "===")
    print(r)
    print(r.source)
    print(r.original_source)
    a_sets = extract_sets(r.original_source)
    print("parsed sets shape:", a_sets.shape)
    print(a_sets)