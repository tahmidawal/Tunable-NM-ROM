"""imp-fronts: run the unchanged select2d.py on an R384-group-only selection result (DESIGN Amendment 9). select2d.main
iterates the registered ranks (128, 256, 384) and indexes the config's group of each; the only change here is that rank
tuple, set to (384,) before main() runs. Everything else (bank binding to inputs/banks/s<label>/BANK.json, target
certification, three-state candidate search, the registered candidate order) is select2d's own code.

    JAX_PLATFORMS=cpu python select2d_r384.py <e2s output dir> <label>     -> checks/select2d-<label>.json
"""
import select2d

select2d.RANKS = (384,)

if __name__ == '__main__':
    select2d.main()
