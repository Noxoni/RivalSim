"""Frozen v0.2 static-world parity tolerances.

These limits were selected only after the measurement-only run completed on
2026-08-23.  The source measurement JSON had SHA-256
7EB62CF97BE25EA5F7CF6540D9D6350829B0AE7887B09F0B63E3915E937B9BDF.

The limits describe acceptable simulator agreement, not an envelope widened
to encompass the observed divergent tails. Contact, sign, and discrete-state
mismatches remain hard failures independent of this table.
"""

V02_PARITY_TOLERANCES: dict[str, float] = {
    "position_uu": 10.0,
    "linear_velocity_uu_per_s": 25.0,
    "orientation_rad": 0.025,
    "angular_velocity_rad_per_s": 0.1,
    "boost": 0.01,
    "handbrake_value": 0.0001,
    "world_contact_normal_rad": 0.05,
}

V02_TOLERANCE_MEASUREMENT_SHA256 = (
    "7EB62CF97BE25EA5F7CF6540D9D6350829B0AE7887B09F0B63E3915E937B9BDF"
)
