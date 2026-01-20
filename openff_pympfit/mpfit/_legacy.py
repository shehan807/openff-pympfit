import numpy as np

def _print_multipole_moments(i, mm, lmax):
    """
    Print multipole moments for site i in a format similar to the original file

    Parameters:
    ----------
    i : int
        Site index
    mm : ndarray
        4D array containing multipole moments
    lmax : int
        Maximum rank for this site
    """
    # Print monopole
    print(f"                   Q00  =  {mm[i, 0, 0, 0]:10.6f}")

    # Print higher order multipoles if present
    for l in range(1, lmax[i] + 1):
        # Calculate and print |Ql|
        q_norm_squared = mm[i, l, 0, 0] ** 2
        for j in range(1, l + 1):
            q_norm_squared += mm[i, l, j, 0] ** 2 + mm[i, l, j, 1] ** 2
        q_norm = np.sqrt(q_norm_squared)

        print(f"|Q{l}| = {q_norm:10.6f}  Q{l}0  = {mm[i, l, 0, 0]:10.6f}", end="")

        # Print components
        for j in range(1, l + 1):
            if j == 1:
                print(
                    f"  Q{l}{j}c = {mm[i, l, j, 0]:10.6f}  Q{l}{j}s = {mm[i, l, j, 1]:10.6f}",
                    end="",
                )
            else:
                # For j > 1, print on new line with spacing
                if j == 2:
                    print()
                print(
                    f"                   Q{l}{j}c = {mm[i, l, j, 0]:10.6f}  Q{l}{j}s = {mm[i, l, j, 1]:10.6f}",
                    end="",
                )
        print()

def numbersites(inpfile):
    count = 0
    with open(inpfile, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_split = line.split()
            if len(line_split) >= 5:
                _type = line_split[0]
                x, y, z = map(float, line_split[1:4])
                maxl = int(line_split[4])
                for i in range(maxl + 1):
                    skip_lines = f.readline()
                count += 1
    return count


def getmultmoments(
    inpfile,
    n,
    lmax,
    mm,  # multipole moments
    ms,  # multipole sites
    atomtype,
    reprint_mm=False,
):
    with open(inpfile, "r") as f:
        for i in range(n):
            line = f.readline().split()
            atomtype[i] = line[0]
            x, y, z = float(line[1]), float(line[2]), float(line[3])
            lmax[i] = int(line[4])

            ms[i, 0] = x
            ms[i, 1] = y
            ms[i, 2] = z

            q0 = float(f.readline().strip())  # monopole
            mm[i, 0, 0, 0] = q0

            if lmax[i] > 0:
                for l in range(1, lmax[i] + 1):
                    line = f.readline().split()
                    mm[i, l, 0, 0] = float(line[0])  # Q_l0
                    for m in range(1, l + 1):  # Q_lm (m>0)
                        idx = 2 * m - 1
                        mm[i, l, m, 0] = float(line[idx])  # real
                        mm[i, l, m, 1] = float(line[idx + 1])  # imaginary

        if reprint_mm:
            # After the with open block:
            for i in range(n):
                print(f"Site {i + 1}:")
                _print_multipole_moments(i, mm, lmax)
                print()
    return lmax, mm, ms, atomtype


def gencharges(ms, qs, midbond):
    """Generate charge positions from multipole sites and bond information"""
    nmult = ms.shape[0]  # number of multipole sites
    nmid = qs.shape[0] - nmult  # number of midpoints

    # copy multipole site coordinates to charge sites
    for i in range(nmult):
        qs[i, 0] = ms[i, 0]
        qs[i, 1] = ms[i, 1]
        qs[i, 2] = ms[i, 2]

    if nmid > 0:
        count = 0
        for i in range(nmult):
            for j in range(i + 1, nmult):
                if midbond[i, j] == 1:
                    # add a midpoint charge
                    qs[nmult + count, 0] = (ms[i, 0] + ms[j, 0]) / 2.0
                    qs[nmult + count, 1] = (ms[i, 1] + ms[j, 1]) / 2.0
                    qs[nmult + count, 2] = (ms[i, 2] + ms[j, 2]) / 2.0
                    count += 1

    return qs
