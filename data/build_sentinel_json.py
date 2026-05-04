import json
from pathlib import Path

import pandas as pd


DEFAULT_CROPS = ("corn", "cotton", "soybean", "winterwheat")


def build_record(crop, year, county_info):
    """Build one Sentinel-2 metadata record for a crop, county, and year."""
    fips = str(county_info["FIPS"])
    state = county_info["State"]
    county = county_info["County"]

    return {
        "FIPS": fips,
        "year": int(year),
        "county": county,
        "state": state,
        "county_ansi": fips[2:],
        "state_ansi": fips[:2],
        "data": {
            "USDA": f"USDA/data/{crop.title()}/{year}/USDA_{crop.title()}_County_{year}.csv",
            "sentinel": [
                f"Sentinel-2 Imagery/data/{year}/{state}/Agriculture_{fips[:2]}_{state}_{year}-04-01_{year}-06-30.h5",
                f"Sentinel-2 Imagery/data/{year}/{state}/Agriculture_{fips[:2]}_{state}_{year}-07-01_{year}-09-30.h5",
            ],
        },
    }


def build_crop_json(
    crop,
    county_info_path=Path("data/county_info.csv"),
    output_root=Path("datasets/sentinel2/json"),
    train_years=(2019, 2020, 2021),
    test_year=2022,
):
    """Build train and test json files for one Sentinel-2 crop."""
    crop = crop.lower()
    county_info = pd.read_csv(county_info_path)

    train_records = []
    test_records = []
    for _, row in county_info.iterrows():
        for year in train_years:
            train_records.append(build_record(crop, year, row))
        test_records.append(build_record(crop, test_year, row))

    crop_output_dir = Path(output_root) / crop
    crop_output_dir.mkdir(parents=True, exist_ok=True)

    train_path = crop_output_dir / f"{crop}_train.json"
    test_path = crop_output_dir / f"{crop}_test.json"
    train_path.write_text(json.dumps(train_records, indent=2))
    test_path.write_text(json.dumps(test_records, indent=2))

    print(f"Saved {len(train_records)} training records to {train_path}")
    print(f"Saved {len(test_records)} test records to {test_path}")


def build_all_crops(
    crops=DEFAULT_CROPS,
    county_info_path=Path("data/county_info.csv"),
    output_root=Path("datasets/sentinel2/json"),
    train_years=(2019, 2020, 2021),
    test_year=2022,
):
    """Build train and test json files for all supported Sentinel-2 crops."""
    for crop in crops:
        build_crop_json(
            crop=crop,
            county_info_path=county_info_path,
            output_root=output_root,
            train_years=train_years,
            test_year=test_year,
        )


if __name__ == "__main__":
    build_all_crops()
