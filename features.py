import numpy as np

num_cols = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

i_rooms      = num_cols.index("total_rooms")
i_bedrooms   = num_cols.index("total_bedrooms")
i_pop        = num_cols.index("population")
i_households = num_cols.index("households")

# Original skewed columns
log_idx = [i_rooms, i_bedrooms, i_pop, i_households]

# Derived ratio features are at indices 8, 9, 10
DERIVED_LOG_OFFSETS = [0, 1, 2]

CITIES = {
    "sf": (37.7749, -122.4194),
    "la": (34.0522, -118.2437),
    "sd": (32.7157, -117.1611),
}


def add_features_np(X_num):

    X_num = X_num.copy()

    rooms      = X_num[:, i_rooms]
    bedrooms   = X_num[:, i_bedrooms]
    pop        = X_num[:, i_pop]
    households = X_num[:, i_households]
    lat        = X_num[:, num_cols.index("latitude")]
    lon        = X_num[:, num_cols.index("longitude")]

    # Safe division
    households_safe = np.where(households == 0, np.nan, households)
    rooms_safe      = np.where(rooms == 0,      np.nan, rooms)

    rooms_per_household      = rooms    / households_safe
    population_per_household = pop      / households_safe
    bedrooms_per_room        = bedrooms / rooms_safe

    distances = []
    for city, (clat, clon) in CITIES.items():
        dist = np.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
        distances.append(dist)

    return np.column_stack([
        X_num,
        rooms_per_household,
        population_per_household,
        bedrooms_per_room,
        *distances,
    ])


def log1p_selected_np(X_num_plus):

    X_num_plus = X_num_plus.copy()
    base_width = len(num_cols)

    # Original skewed columns
    for idx in log_idx:
        X_num_plus[:, idx] = np.log1p(np.clip(X_num_plus[:, idx], 0, None))

    # Derived ratio features only (NOT distances)
    for offset in DERIVED_LOG_OFFSETS:
        col = base_width + offset
        X_num_plus[:, col] = np.log1p(np.clip(X_num_plus[:, col], 0, None))

    return X_num_plus

dist_sf: [0.1, 0.5, 1.2, 3.4, 8.9]

rooms: [100, 500, 2000, 8000]   