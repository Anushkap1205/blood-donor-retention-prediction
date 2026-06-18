#!/usr/bin/env python3
"""Open research papers in the default web browser."""

import webbrowser

PAPERS = {
    "1. Yeh et al. (2009) - RFMTC Base": "https://doi.org/10.1016/j.eswa.2008.12.015",
    "2. Kauten et al. (2021) - ML Pipeline": "https://doi.org/10.1007/s10796-020-10041-0",
    "3. Yang et al. (2020) - Reactivation RCT": "https://doi.org/10.1186/s12889-020-09347-w",
    "4. Liu et al. (2022) - COVID SMS predictions": "https://doi.org/10.1371/journal.pone.0263654",
    "5. Marwaha et al. (2012) - India Deferral Rates": "https://doi.org/10.4103/0973-6247.95048",
    "6. Bagot et al. (2016) - First-Time Retention": "https://doi.org/10.1016/j.tmrv.2016.05.006",
}


def main():
    print("Select a research paper to open in your browser:")
    keys = list(PAPERS.keys())
    for index, name in enumerate(keys, 1):
        print(f"[{index}] {name}")
    print(f"[{len(keys) + 1}] Open All (6 papers)")

    try:
        choice = input("Enter choice number: ").strip()
        if not choice:
            return

        choice_idx = int(choice)
        if choice_idx == len(keys) + 1:
            print("Opening all links...")
            for url in PAPERS.values():
                webbrowser.open(url)
        elif 1 <= choice_idx <= len(keys):
            name = keys[choice_idx - 1]
            url = PAPERS[name]
            print(f"Opening link for {name}...")
            webbrowser.open(url)
        else:
            print("Invalid index.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
