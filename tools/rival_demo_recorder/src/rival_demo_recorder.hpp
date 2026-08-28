#pragma once

#include "rivalrec/recorder_policy.hpp"
#include "rivalrec/recording_format.hpp"

#include "bakkesmod/plugin/bakkesmodplugin.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

struct ControllerInput;
class ActorWrapper;
class CarWrapper;
class PriWrapper;
class ServerWrapper;
class VehiclePickupWrapper;

class RivalDemoRecorder final : public BakkesMod::Plugin::BakkesModPlugin {
public:
    enum class SessionType { match, freeplay };

    void onLoad() override;
    void onUnload() override;

private:
    enum class Journal { events, markers };

    struct JournalRecord {
        Journal journal{};
        std::string json;
    };

    using WriteItem = std::variant<rivalrec::Frame, JournalRecord>;

    struct SessionConfig {
        SessionType type{};
        std::string label;
        std::string opponent_label;
        std::string mechanic_label;
        rivalrec::SessionUuid uuid;
        std::filesystem::path directory;
        std::string capture_start_utc;
        std::string capture_start_local;
        std::string rocket_league_build;
        int bakkesmod_version{};
        float engine_physics_framerate{};
        std::string map;
        std::string local_player_id;
        std::string local_player_name;
        std::int32_t local_player_numeric_id{-1};
        std::int8_t local_team{-1};
        std::uint32_t chunk_frames{3600};
        std::uint32_t flush_frames{120};
        std::uint32_t max_queue_frames{4096};
    };

    struct CaptureCounters {
        std::atomic<std::uint64_t> attempted_frames{};
        std::atomic<std::uint64_t> enqueued_frames{};
        std::atomic<std::uint64_t> written_frames{};
        std::atomic<std::uint64_t> queue_dropped_frames{};
        std::atomic<std::uint64_t> duplicate_physics_frames{};
        std::atomic<std::uint64_t> duplicate_hook_callbacks{};
        std::atomic<std::uint64_t> duplicate_frames_suppressed{};
        std::atomic<std::uint64_t> duplicate_frames_retained{};
        std::atomic<std::uint64_t> out_of_order_physics_frames{};
        std::atomic<std::uint64_t> missing_physics_frames{};
        std::atomic<std::uint64_t> identity_failures{};
        std::atomic<std::uint64_t> local_car_rebinds{};
        std::atomic<std::uint64_t> event_records{};
        std::atomic<std::uint64_t> marker_records{};
        std::atomic<std::int32_t> first_frame_physics{-1};
        std::atomic<std::int32_t> last_frame_physics{-1};
        std::atomic<float> first_frame_engine_time{};
        std::atomic<float> last_frame_engine_time{};
        std::atomic<std::int32_t> stop_physics_frame{-1};
        std::atomic<float> stop_engine_time{};
        std::atomic<std::uint64_t> stop_monotonic_ns{};
    };

    struct LocalHumanIdentity {
        std::string stable_id;
        std::string player_name;
        std::int32_t player_id{-1};
        std::int8_t team{-1};
        std::uintptr_t car_address{};
    };

    struct PadObservation {
        rivalrec::BoostPadState state;
        float picked_up_game_time{};
    };

    void register_commands();
    void register_hooks();
    void start_command(const std::vector<std::string>& args);
    void stop_command(const std::vector<std::string>& args);
    void status_command(const std::vector<std::string>& args) const;
    void mark_command(const std::vector<std::string>& args);
    void note_command(const std::vector<std::string>& args);

    void on_vehicle_input(CarWrapper caller, void* params, const std::string& event_name);
    void on_car_event(CarWrapper caller, void* params, const std::string& event_name);
    void on_server_event(ServerWrapper caller, void* params, const std::string& event_name);
    void on_actor_event(ActorWrapper caller, void* params, const std::string& event_name);
    void on_pickup_event(ActorWrapper caller, void* params, const std::string& event_name,
                         bool active);

    [[nodiscard]] bool allowed_capture_context(std::string& reason) const;
    [[nodiscard]] bool resolve_unique_local_human(LocalHumanIdentity& identity,
                                                  std::string& reason) const;
    [[nodiscard]] rivalrec::Frame capture_frame(const ControllerInput& input,
                                                std::uintptr_t local_car_address);
    [[nodiscard]] rivalrec::BallState capture_ball(ServerWrapper server) const;
    [[nodiscard]] rivalrec::CarState capture_car(PriWrapper pri,
                                                 std::uintptr_t local_car_address) const;
    [[nodiscard]] rivalrec::MatchState capture_match(ServerWrapper server) const;
    [[nodiscard]] std::vector<rivalrec::BoostPadState> snapshot_pads(float game_time);

    [[nodiscard]] bool enqueue(WriteItem item, bool is_frame);
    void publish_frame(rivalrec::Frame frame);
    void flush_pending_frame();
    void remember_rebind_context(const std::string& context);
    void record_rebind_event(std::uintptr_t previous_address,
                             const LocalHumanIdentity& identity,
                             const std::string& event_context);
    void record_duplicate_callback_event(const rivalrec::Frame& suppressed,
                                         const rivalrec::Frame& retained);
    void writer_main(SessionConfig config);
    void stop_session(bool clean, const std::string& reason);
    void write_manifest(const SessionConfig& config, bool clean, const std::string& reason,
                        const std::vector<rivalrec::ChunkInfo>& chunks,
                        const std::vector<rivalrec::FileInfo>& journals,
                        const std::string& capture_end_utc) const;
    void record_event(const std::string& kind, const std::string& unreal_event,
                      std::uintptr_t caller, const std::string& actor_id = "",
                      const std::string& extra_json = "");
    void record_marker(const std::string& kind, const std::string& text);
    [[nodiscard]] std::string journal_prefix() const;
    [[nodiscard]] std::string configured_data_root() const;
    [[nodiscard]] std::string join_arguments(const std::vector<std::string>& args,
                                             std::size_t start) const;
    void log(const std::string& message) const;
    void error(const std::string& message) const;

    std::atomic<bool> recording_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> writer_failed_{false};
    mutable std::mutex session_mutex_;
    std::optional<SessionConfig> session_;
    std::uintptr_t local_car_address_{};
    std::string last_rebind_context_;
    std::chrono::steady_clock::time_point session_steady_start_{};
    std::string stop_reason_;
    bool clean_stop_{};

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<WriteItem> queue_;
    std::thread writer_thread_;
    CaptureCounters counters_;

    mutable std::mutex capture_mutex_;
    rivalrec::TickFrameReducer tick_reducer_;

    mutable std::mutex pads_mutex_;
    std::map<std::uintptr_t, PadObservation> pads_;
};
