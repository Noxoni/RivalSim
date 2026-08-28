#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rivalrec {

inline constexpr std::uint32_t kSchemaVersion = 1;
inline constexpr char kSchemaName[] = "RIVALRL_NATIVE_DEMO_V1";

struct Vec3 {
    float x{};
    float y{};
    float z{};
};

struct Rotator {
    std::int32_t pitch{};
    std::int32_t yaw{};
    std::int32_t roll{};
};

struct NativeInput {
    float throttle{};
    float steer{};
    float pitch{};
    float yaw{};
    float roll{};
    float dodge_forward{};
    float dodge_strafe{};
    bool handbrake{};
    bool jump{};
    bool activate_boost{};
    bool holding_boost{};
    bool jumped{};
};

struct RivalAction {
    float throttle{};
    float steer{};
    float pitch{};
    float yaw{};
    float roll{};
    bool jump{};
    bool boost{};
    bool handbrake{};
};

struct ComponentState {
    bool active{};
    float activity_time{};
};

struct WheelState {
    std::int8_t index{-1};
    bool has_contact{};
    bool has_world_contact{};
    float contact_change_time{};
    Vec3 contact_location{};
    Vec3 contact_normal{};
    Vec3 lateral_direction{};
    Vec3 longitudinal_direction{};
    Vec3 reference_location{};
    float suspension_distance{};
    float spin_speed{};
};

struct CarFlags {
    bool is_local_human{};
    bool is_bot{};
    bool car_present{};
    bool demolished{};
    bool on_ground{};
    bool supersonic{};
    bool jumped{};
    bool double_jumped{};
    bool can_jump{};
    bool has_flip{};
};

struct CarState {
    std::string stable_id;
    std::string player_name;
    std::int32_t player_id{-1};
    std::int8_t team{-1};
    CarFlags flags{};
    std::uint32_t availability{};
    Vec3 position{};
    Rotator rotation{};
    Vec3 linear_velocity{};
    Vec3 angular_velocity{};
    float boost{};
    float time_off_ground{};
    float time_on_ground{};
    std::int32_t last_ball_touch_frame{-1};
    std::int32_t last_ball_impact_frame{-1};
    std::int32_t respawn_time_remaining{};
    std::int16_t num_wheel_world_contacts{};
    std::int16_t num_wheel_contacts{};
    ComponentState boost_component{};
    ComponentState jump_component{};
    ComponentState double_jump_component{};
    ComponentState dodge_component{};
    Vec3 dodge_direction{};
    ComponentState flip_component{};
    float flip_time{};
    bool flip_right{};
    bool native_input_available{};
    NativeInput native_input{};
    std::vector<WheelState> wheels;
};

struct BallState {
    std::uint32_t availability{};
    Vec3 position{};
    Rotator rotation{};
    Vec3 linear_velocity{};
    Vec3 angular_velocity{};
    float gravity_z{};
    float gravity_scale{};
    float last_touch_time{};
    float last_hit_world_time{};
    std::int8_t hit_team{-1};
    std::string current_affector_id;
};

struct MatchFlags {
    bool paused{};
    bool overtime{};
    bool round_active{};
    bool match_ended{};
    bool ball_has_been_hit{};
    bool kickoff_or_countdown{};
};

struct MatchState {
    std::string game_mode;
    std::string map;
    std::string match_guid;
    float seconds_elapsed{};
    float seconds_remaining{};
    float total_game_time_played{};
    float overtime_time_played{};
    std::int32_t score_team_0{};
    std::int32_t score_team_1{};
    std::int32_t round_number{};
    std::int32_t countdown_number{};
    std::int32_t seconds_remaining_countdown{};
    MatchFlags flags{};
    std::uint32_t availability{};
};

struct BoostPadState {
    std::string stable_id;
    Vec3 position{};
    bool is_full_boost{};
    bool active{};
    bool picked_up{};
    std::uint8_t cooldown_quality{};
    std::uint8_t boost_type{};
    float boost_amount{};
    float respawn_delay{};
    float cooldown_remaining{};
};

struct Frame {
    std::uint64_t sequence{};
    std::int32_t physics_frame{};
    std::int32_t replicated_physics_frame{};
    float engine_physics_time{};
    float game_time_seconds{};
    std::uint64_t monotonic_ns{};
    std::uint64_t utc_ns{};
    std::uint64_t delta_monotonic_ns{};
    float delta_engine_seconds{};
    std::uint32_t status_flags{};
    std::uint32_t missing_physics_frames{};
    NativeInput native_input{};
    RivalAction rival_action{};
    std::optional<BallState> ball;
    std::vector<CarState> cars;
    MatchState match{};
    std::vector<BoostPadState> boost_pads;
};

struct SessionUuid {
    std::array<std::uint8_t, 16> bytes{};
    std::string text;
};

struct ChunkInfo {
    std::uint32_t index{};
    std::string relative_path;
    std::uint64_t frame_count{};
    std::uint64_t first_sequence{};
    std::uint64_t last_sequence{};
    std::uint64_t bytes{};
    std::string sha256;
    bool complete{};
};

struct FileInfo {
    std::string relative_path;
    std::uint64_t bytes{};
    std::string sha256;
};

std::vector<std::uint8_t> encode_frame(const Frame& frame);
std::uint32_t crc32(const std::vector<std::uint8_t>& bytes);
std::string sha256_file(const std::filesystem::path& path);
FileInfo inspect_file(const std::filesystem::path& session_directory,
                      const std::filesystem::path& path);
SessionUuid make_session_uuid();
std::string json_escape(const std::string& value);
std::string utc_now_iso8601();
std::uint64_t utc_now_ns();
void atomic_write_text(const std::filesystem::path& path, const std::string& text);
void durable_flush_file(const std::filesystem::path& path);

class ChunkWriter {
public:
    ChunkWriter(
        const std::filesystem::path& session_directory,
        std::uint32_t chunk_index,
        const SessionUuid& session_uuid,
        std::uint64_t first_sequence,
        std::uint32_t flush_frames);
    ChunkWriter(const ChunkWriter&) = delete;
    ChunkWriter& operator=(const ChunkWriter&) = delete;
    ~ChunkWriter();

    void append(const Frame& frame);
    [[nodiscard]] bool full(std::uint32_t chunk_frames) const;
    [[nodiscard]] std::uint64_t frame_count() const;
    ChunkInfo close();
    void abandon();

private:
    void write_record(std::uint8_t type, const std::vector<std::uint8_t>& payload);

    std::filesystem::path session_directory_;
    std::filesystem::path partial_path_;
    std::filesystem::path final_path_;
    std::ofstream stream_;
    std::uint32_t chunk_index_{};
    std::uint32_t flush_frames_{};
    std::uint64_t frame_count_{};
    std::uint64_t first_sequence_{};
    std::uint64_t last_sequence_{};
    bool closed_{};
};

}  // namespace rivalrec
