#include "rival_demo_recorder.hpp"

#include "bakkesmod/wrappers/Engine/EngineTAWrapper.h"
#include "bakkesmod/wrappers/Engine/UnrealStringWrapper.h"
#include "bakkesmod/wrappers/Engine/WorldInfoWrapper.h"
#include "bakkesmod/wrappers/GameEvent/ServerWrapper.h"
#include "bakkesmod/wrappers/GameObject/BallWrapper.h"
#include "bakkesmod/wrappers/GameObject/BoostPickupWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/BoostWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/CarComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/DodgeComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/DoubleJumpComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/FlipCarComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/JumpComponentWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/VehicleSimWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarComponent/WheelWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarWrapper.h"
#include "bakkesmod/wrappers/GameObject/PriWrapper.h"
#include "bakkesmod/wrappers/GameObject/TeamWrapper.h"
#include "bakkesmod/wrappers/GameObject/VehiclePickupWrapper.h"
#include "bakkesmod/wrappers/PlayerControllerWrapper.h"
#include "bakkesmod/wrappers/arraywrapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <memory>
#include <cstring>
#include <sstream>
#include <stdexcept>

#ifndef RIVALREC_GIT_SHA
#define RIVALREC_GIT_SHA "unknown"
#endif
#ifndef RIVALREC_SDK_REVISION
#define RIVALREC_SDK_REVISION "unknown"
#endif
#ifndef RIVALREC_PLUGIN_BUILD
#define RIVALREC_PLUGIN_BUILD "dev"
#endif

BAKKESMOD_PLUGIN(RivalDemoRecorder, "Rival 2.0 Human Demonstration Recorder",
                 RIVALREC_PLUGIN_BUILD, PLUGINTYPE_FREEPLAY | PLUGINTYPE_CUSTOM_TRAINING)

namespace {

constexpr std::uint32_t kFrameDuplicate = 1U << 0U;
constexpr std::uint32_t kFrameOutOfOrder = 1U << 1U;
constexpr std::uint32_t kFrameGap = 1U << 2U;
constexpr std::uint32_t kBallTransform = 1U << 0U;
constexpr std::uint32_t kBallDynamics = 1U << 1U;
constexpr std::uint32_t kBallGravity = 1U << 2U;
constexpr std::uint32_t kBallTouch = 1U << 3U;
constexpr std::uint32_t kBallAffector = 1U << 4U;

constexpr std::uint32_t kCarIdentity = 1U << 0U;
constexpr std::uint32_t kCarTransform = 1U << 1U;
constexpr std::uint32_t kCarDynamics = 1U << 2U;
constexpr std::uint32_t kCarBoost = 1U << 3U;
constexpr std::uint32_t kCarContactFlags = 1U << 4U;
constexpr std::uint32_t kCarJumpFlags = 1U << 5U;
constexpr std::uint32_t kCarComponents = 1U << 6U;
constexpr std::uint32_t kCarInput = 1U << 7U;
constexpr std::uint32_t kCarWheels = 1U << 8U;
constexpr std::uint32_t kCarLifecycle = 1U << 9U;

constexpr std::uint32_t kMatchIdentity = 1U << 0U;
constexpr std::uint32_t kMatchClock = 1U << 1U;
constexpr std::uint32_t kMatchFlags = 1U << 2U;
constexpr std::uint32_t kMatchScore = 1U << 3U;

rivalrec::Vec3 vector(const Vector& value) { return {value.X, value.Y, value.Z}; }

rivalrec::Rotator rotation(const Rotator& value) {
    return {value.Pitch, value.Yaw, value.Roll};
}

rivalrec::NativeInput native_input(const ControllerInput& value) {
    return {.throttle = value.Throttle,
            .steer = value.Steer,
            .pitch = value.Pitch,
            .yaw = value.Yaw,
            .roll = value.Roll,
            .dodge_forward = value.DodgeForward,
            .dodge_strafe = value.DodgeStrafe,
            .handbrake = value.Handbrake != 0U,
            .jump = value.Jump != 0U,
            .activate_boost = value.ActivateBoost != 0U,
            .holding_boost = value.HoldingBoost != 0U,
            .jumped = value.Jumped != 0U};
}

rivalrec::RivalAction rival_action(const ControllerInput& value) {
    return {.throttle = value.Throttle,
            .steer = value.Steer,
            .pitch = value.Pitch,
            .yaw = value.Yaw,
            .roll = value.Roll,
            .jump = value.Jump != 0U,
            .boost = value.ActivateBoost != 0U || value.HoldingBoost != 0U,
            .handbrake = value.Handbrake != 0U};
}

std::uint64_t monotonic_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

std::string local_now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_s(&local, &seconds);
    std::ostringstream output;
    output << std::put_time(&local, "%Y-%m-%dT%H:%M:%S%z");
    return output.str();
}

std::string pointer_id(std::uintptr_t address) {
    std::ostringstream output;
    output << "0x" << std::hex << address;
    return output.str();
}

std::string session_type_name(RivalDemoRecorder::SessionType type) {
    return type == RivalDemoRecorder::SessionType::match ? "match" : "freeplay";
}

std::string safe_pri_id(PriWrapper pri) {
    if (!pri) {
        return {};
    }
    try {
        const std::string id = pri.GetUniqueIdWrapper().GetIdString();
        if (!id.empty()) {
            return id;
        }
    } catch (...) {
    }
    return "pri:" + std::to_string(pri.GetPlayerID());
}

rivalrec::ComponentState component(CarComponentWrapper value) {
    if (!value) {
        return {};
    }
    return {.active = value.GetbActive() != 0U, .activity_time = value.GetActivityTime()};
}

std::string bool_json(bool value) { return value ? "true" : "false"; }

struct OnHitBallParams {
    std::uintptr_t ball;
    Vector hit_location;
    Vector hit_normal;
};

bool finite(const Vector& value) {
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}

}  // namespace

void RivalDemoRecorder::onLoad() {
    register_commands();
    register_hooks();
    log("loaded read-only recorder build " RIVALREC_PLUGIN_BUILD " (SDK "
        RIVALREC_SDK_REVISION ")");
}

void RivalDemoRecorder::onUnload() { stop_session(true, "plugin_unload"); }

void RivalDemoRecorder::register_commands() {
    cvarManager->registerCvar(
        "rivalrec_data_dir", "",
        "Output root. Empty uses BakkesMod data/rival2/human_demos.", true, false, 0.0F,
        false, 0.0F, true);
    cvarManager->registerCvar("rivalrec_chunk_frames", "3600",
                              "Frames per lossless chunk (default 30 seconds at 120 Hz).",
                              true, true, 120.0F, true, 432000.0F, true);
    cvarManager->registerCvar("rivalrec_flush_frames", "120",
                              "Flush interval in frames.", true, true, 1.0F, true,
                              3600.0F, true);
    cvarManager->registerCvar("rivalrec_max_queue_frames", "4096",
                              "Bounded background-writer queue capacity.", true, true,
                              120.0F, true, 65536.0F, true);

    cvarManager->registerNotifier(
        "rivalrec_start", [this](const std::vector<std::string>& args) { start_command(args); },
        "Start read-only recording: rivalrec_start [match|freeplay] [label]", 0);
    cvarManager->registerNotifier(
        "rivalrec_stop", [this](const std::vector<std::string>& args) { stop_command(args); },
        "Stop and finalize the current recording", 0);
    cvarManager->registerNotifier(
        "rivalrec_status", [this](const std::vector<std::string>& args) { status_command(args); },
        "Print recorder status and integrity counters", 0);
    cvarManager->registerNotifier(
        "rivalrec_mark", [this](const std::vector<std::string>& args) { mark_command(args); },
        "Add optional metadata marker: rivalrec_mark <label>", 0);
    cvarManager->registerNotifier(
        "rivalrec_note", [this](const std::vector<std::string>& args) { note_command(args); },
        "Add arbitrary session note: rivalrec_note <text>", 0);
}

void RivalDemoRecorder::register_hooks() {
    gameWrapper->HookEventWithCaller<CarWrapper>(
        "Function TAGame.Car_TA.SetVehicleInput",
        [this](CarWrapper caller, void* params, std::string event_name) {
            on_vehicle_input(caller, params, event_name);
        });

    const std::vector<std::string> car_events = {
        "Function TAGame.Car_TA.OnHitBall",
        "Function TAGame.Car_TA.HandleWheelBallHit",
        "Function TAGame.Car_TA.OnJumpPressed",
        "Function TAGame.Car_TA.Demolish2",
        "Function TAGame.Car_TA.DemolishDestroyTimer",
        "Function TAGame.Car_TA.RespawnInPlace",
    };
    for (const std::string& event : car_events) {
        gameWrapper->HookEventWithCaller<CarWrapper>(
            event, [this](CarWrapper caller, void* params, std::string event_name) {
                on_car_event(caller, params, event_name);
            });
    }

    gameWrapper->HookEventWithCaller<CarComponentWrapper>(
        "Function TAGame.DodgeComponent_TA.ApplyDodgeImpulse",
        [this](CarComponentWrapper caller, void*, std::string event_name) {
            CarWrapper car = caller ? caller.GetCar() : CarWrapper(0);
            record_event("dodge_or_flip_onset", event_name,
                         caller ? caller.memory_address : 0,
                         car ? safe_pri_id(car.GetPRI()) : "");
        });

    const std::vector<std::string> server_events = {
        "Function TAGame.GameEvent_Soccar_TA.EventGoalScored",
        "Function TAGame.GameEvent_Soccar_TA.StartNewRound",
        "Function TAGame.GameEvent_Soccar_TA.ResetGame",
        "Function TAGame.GameEvent_TA.OnCarSpawned",
        "Function TAGame.GameEvent_TA.EventPlayerResetTraining",
    };
    for (const std::string& event : server_events) {
        gameWrapper->HookEventWithCaller<ServerWrapper>(
            event, [this](ServerWrapper caller, void* params, std::string event_name) {
                on_server_event(caller, params, event_name);
            });
    }

    gameWrapper->HookEventWithCaller<ActorWrapper>(
        "Function TAGame.VehiclePickup_TA.EventPickedUp",
        [this](ActorWrapper caller, void* params, std::string event_name) {
            on_pickup_event(caller, params, event_name, false);
        });
    gameWrapper->HookEventWithCaller<ActorWrapper>(
        "Function TAGame.VehiclePickup_TA.EventSpawned",
        [this](ActorWrapper caller, void* params, std::string event_name) {
            on_pickup_event(caller, params, event_name, true);
        });
    gameWrapper->HookEventWithCaller<ActorWrapper>(
        "Function TAGame.Ball_TA.FellOutOfWorld",
        [this](ActorWrapper caller, void* params, std::string event_name) {
            on_actor_event(caller, params, event_name);
        });
}

void RivalDemoRecorder::start_command(const std::vector<std::string>& args) {
    if (recording_.load()) {
        error("already recording; run rivalrec_stop first");
        return;
    }
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
    std::string context_reason;
    if (!allowed_capture_context(context_reason)) {
        error("start rejected: " + context_reason);
        return;
    }

    SessionConfig config;
    const std::string requested = args.size() > 1 ? args[1] : "";
    if (requested.empty()) {
        config.type = gameWrapper->IsInFreeplay() ? SessionType::freeplay : SessionType::match;
    } else if (requested == "match") {
        config.type = SessionType::match;
    } else if (requested == "freeplay") {
        config.type = SessionType::freeplay;
    } else {
        error("first argument must be 'match' or 'freeplay'");
        return;
    }
    config.label = join_arguments(args, 2);
    if (config.type == SessionType::match) {
        config.opponent_label = config.label;
    } else {
        config.mechanic_label = config.label;
    }
    if ((config.type == SessionType::freeplay) != gameWrapper->IsInFreeplay()) {
        error("requested session type does not match the active Rocket League context");
        return;
    }

    std::uintptr_t local_address = 0;
    std::string identity_reason;
    if (!resolve_unique_local_human(&config, &local_address, identity_reason)) {
        error("cannot uniquely identify the local human car: " + identity_reason);
        return;
    }

    config.uuid = rivalrec::make_session_uuid();
    config.capture_start_utc = rivalrec::utc_now_iso8601();
    config.capture_start_local = local_now_iso8601();
    config.rocket_league_build = gameWrapper->GetPsyBuildID();
    config.bakkesmod_version = gameWrapper->GetBakkesModVersion();
    EngineTAWrapper engine = gameWrapper->GetEngine();
    config.engine_physics_framerate = engine ? engine.GetPhysicsFramerate() : 0.0F;
    config.map = gameWrapper->GetCurrentMap();
    config.chunk_frames = static_cast<std::uint32_t>(
        cvarManager->getCvar("rivalrec_chunk_frames").getIntValue());
    config.flush_frames = static_cast<std::uint32_t>(
        cvarManager->getCvar("rivalrec_flush_frames").getIntValue());
    config.max_queue_frames = static_cast<std::uint32_t>(
        cvarManager->getCvar("rivalrec_max_queue_frames").getIntValue());
    config.directory = std::filesystem::path(configured_data_root()) / config.uuid.text;
    std::filesystem::create_directories(config.directory / "chunks");

    counters_.attempted_frames = 0;
    counters_.enqueued_frames = 0;
    counters_.written_frames = 0;
    counters_.queue_dropped_frames = 0;
    counters_.duplicate_physics_frames = 0;
    counters_.out_of_order_physics_frames = 0;
    counters_.missing_physics_frames = 0;
    counters_.identity_failures = 0;
    counters_.event_records = 0;
    counters_.marker_records = 0;
    next_sequence_ = 0;
    previous_physics_frame_ = -1;
    previous_engine_time_ = 0.0F;
    previous_monotonic_ns_ = 0;
    session_steady_start_ = std::chrono::steady_clock::now();
    stop_reason_.clear();
    clean_stop_ = false;
    writer_failed_ = false;
    stop_requested_ = false;
    {
        std::lock_guard queue_lock(queue_mutex_);
        queue_.clear();
    }
    {
        std::lock_guard pad_lock(pads_mutex_);
        pads_.clear();
    }
    {
        std::lock_guard session_lock(session_mutex_);
        session_ = config;
        local_car_address_ = local_address;
    }
    recording_ = true;
    writer_thread_ = std::thread(&RivalDemoRecorder::writer_main, this, config);
    log("recording " + session_type_name(config.type) + " session " + config.uuid.text +
        " at " + config.directory.string());
}

void RivalDemoRecorder::stop_command(const std::vector<std::string>&) {
    stop_session(true, "user_stop");
}

void RivalDemoRecorder::status_command(const std::vector<std::string>&) const {
    if (!recording_.load()) {
        log(writer_failed_.load() ? "not recording; last writer failed" : "not recording");
        return;
    }
    std::optional<SessionConfig> config;
    {
        std::lock_guard lock(session_mutex_);
        config = session_;
    }
    std::size_t queued = 0;
    {
        std::lock_guard lock(queue_mutex_);
        queued = queue_.size();
    }
    log("recording session=" + (config ? config->uuid.text : "unknown") +
        " attempted=" + std::to_string(counters_.attempted_frames.load()) +
        " written=" + std::to_string(counters_.written_frames.load()) +
        " queued=" + std::to_string(queued) +
        " queue_dropped=" + std::to_string(counters_.queue_dropped_frames.load()) +
        " duplicate_physics=" +
        std::to_string(counters_.duplicate_physics_frames.load()) +
        " out_of_order=" + std::to_string(counters_.out_of_order_physics_frames.load()) +
        " missing_physics=" + std::to_string(counters_.missing_physics_frames.load()));
}

void RivalDemoRecorder::mark_command(const std::vector<std::string>& args) {
    const std::string text = join_arguments(args, 1);
    if (text.empty()) {
        error("usage: rivalrec_mark <label>");
        return;
    }
    record_marker("marker", text);
}

void RivalDemoRecorder::note_command(const std::vector<std::string>& args) {
    const std::string text = join_arguments(args, 1);
    if (text.empty()) {
        error("usage: rivalrec_note <text>");
        return;
    }
    record_marker("note", text);
}

void RivalDemoRecorder::on_vehicle_input(CarWrapper caller, void* params,
                                         const std::string&) {
    if (!recording_.load() || params == nullptr || !caller) {
        return;
    }
    std::uintptr_t local_address = 0;
    {
        std::lock_guard lock(session_mutex_);
        local_address = local_car_address_;
    }
    if (caller.memory_address != local_address) {
        return;
    }

    // This is the only access to the hook parameter: copy the native value exactly.
    // The recorder never writes to params and never calls any gameplay input setter.
    const ControllerInput input = *static_cast<const ControllerInput*>(params);
    ++counters_.attempted_frames;
    try {
        std::string reason;
        std::uintptr_t resolved_address = 0;
        if (!resolve_unique_local_human(nullptr, &resolved_address, reason) ||
            resolved_address != local_address) {
            ++counters_.identity_failures;
            stop_session(false, "local_human_identity_lost:" + reason);
            error("recording stopped: local human car identity became ambiguous");
            return;
        }
        rivalrec::Frame frame = capture_frame(input, local_address);
        if (enqueue(std::move(frame), true)) {
            ++counters_.enqueued_frames;
        }
    } catch (const std::exception& exception) {
        stop_session(false, "capture_exception:" + std::string(exception.what()));
        error("recording stopped after capture exception: " + std::string(exception.what()));
    }
}

void RivalDemoRecorder::on_car_event(CarWrapper caller, void* params,
                                     const std::string& event_name) {
    std::string actor_id;
    if (caller) {
        actor_id = safe_pri_id(caller.GetPRI());
    }
    std::string kind = "car_event";
    if (event_name.find("OnHitBall") != std::string::npos) {
        kind = "ball_touch";
    } else if (event_name.find("WheelBallHit") != std::string::npos) {
        kind = "wheel_ball_contact";
    } else if (event_name.find("JumpPressed") != std::string::npos) {
        kind = "jump_onset";
    } else if (event_name.find("Demolish2") != std::string::npos) {
        kind = "demolition";
    } else if (event_name.find("DemolishDestroyTimer") != std::string::npos) {
        kind = "demolished_car_removed";
    } else if (event_name.find("Respawn") != std::string::npos) {
        kind = "respawn";
    }
    std::string extra_json;
    if (event_name.find("OnHitBall") != std::string::npos && params != nullptr) {
        OnHitBallParams hit{};
        std::memcpy(&hit, params, sizeof(hit));
        if (finite(hit.hit_location) && finite(hit.hit_normal)) {
            std::ostringstream extra;
            extra << "\"contacting_car_id\":\"" << rivalrec::json_escape(actor_id)
                  << "\",\"ball_address\":\"" << pointer_id(hit.ball)
                  << "\",\"hit_location\":[" << hit.hit_location.X << ','
                  << hit.hit_location.Y << ',' << hit.hit_location.Z
                  << "],\"hit_normal\":[" << hit.hit_normal.X << ',' << hit.hit_normal.Y
                  << ',' << hit.hit_normal.Z << ']';
            extra_json = extra.str();
        }
    } else if (event_name.find("WheelBallHit") != std::string::npos && params != nullptr) {
        std::uintptr_t wheel_address = 0;
        std::memcpy(&wheel_address, params, sizeof(wheel_address));
        extra_json = "\"wheel_address\":\"" + pointer_id(wheel_address) + "\"";
    }
    record_event(kind, event_name, caller ? caller.memory_address : 0, actor_id,
                 extra_json);
}

void RivalDemoRecorder::on_server_event(ServerWrapper caller, void*,
                                        const std::string& event_name) {
    std::string kind = "match_event";
    if (event_name.find("GoalScored") != std::string::npos) {
        kind = "goal";
    } else if (event_name.find("StartNewRound") != std::string::npos) {
        kind = "kickoff_or_round_reset";
    } else if (event_name.find("ResetGame") != std::string::npos) {
        kind = "game_reset";
    } else if (event_name.find("OnCarSpawned") != std::string::npos) {
        kind = "respawn";
    } else if (event_name.find("PlayerResetTraining") != std::string::npos) {
        kind = "freeplay_reset";
    }
    record_event(kind, event_name, caller ? caller.memory_address : 0);
}

void RivalDemoRecorder::on_actor_event(ActorWrapper caller, void*,
                                       const std::string& event_name) {
    const std::string kind = event_name.find("FellOutOfWorld") != std::string::npos
                                 ? "ball_world_out_of_bounds"
                                 : "actor_event";
    record_event(kind, event_name, caller ? caller.memory_address : 0);
}

void RivalDemoRecorder::on_pickup_event(ActorWrapper caller, void*,
                                        const std::string& event_name, bool active) {
    if (!recording_.load() || !caller) {
        return;
    }
    try {
        BoostPickupWrapper pickup(caller.memory_address);
        PadObservation observation;
        observation.state.stable_id = "pickup:" + pointer_id(caller.memory_address);
        observation.state.position = vector(pickup.GetLocation());
        observation.state.boost_amount = pickup.GetBoostAmount();
        observation.state.boost_type = pickup.GetBoostType();
        observation.state.is_full_boost = observation.state.boost_type != 0U;
        observation.state.active = active;
        observation.state.picked_up = !active;
        observation.state.respawn_delay = pickup.GetRespawnDelay();
        observation.state.cooldown_quality = 1;  // event-timed derivation, not native timer
        ServerWrapper server = gameWrapper->GetGameEventAsServer();
        observation.picked_up_game_time = (!active && server) ? server.GetSecondsElapsed() : 0.0F;
        {
            std::lock_guard lock(pads_mutex_);
            pads_[caller.memory_address] = observation;
        }
        record_event(active ? "boost_pad_spawn" : "boost_pad_pickup", event_name,
                     caller.memory_address, observation.state.stable_id);
    } catch (const std::exception& exception) {
        record_event("boost_pad_event_unreadable", event_name, caller.memory_address,
                     exception.what());
    }
}

bool RivalDemoRecorder::allowed_capture_context(std::string& reason) const {
    if (!gameWrapper->IsInGame()) {
        reason = "Rocket League is not in a game";
        return false;
    }
    if (gameWrapper->IsInOnlineGame()) {
        reason = "online games are intentionally out of scope";
        return false;
    }
    ServerWrapper server = gameWrapper->GetGameEventAsServer();
    if (!server) {
        reason = "no local server game event is available";
        return false;
    }
    if (!gameWrapper->IsInFreeplay() && !server.IsPlayingOffline()) {
        reason = "only Freeplay, offline/local, and RLBot-created local games are allowed";
        return false;
    }
    return true;
}

bool RivalDemoRecorder::resolve_unique_local_human(SessionConfig* metadata,
                                                   std::uintptr_t* local_car_address,
                                                   std::string& reason) const {
    CarWrapper local_car = gameWrapper->GetLocalCar();
    PlayerControllerWrapper controller = gameWrapper->GetPlayerController();
    if (!local_car || !controller) {
        reason = "GetLocalCar or GetPlayerController returned null";
        return false;
    }
    PriWrapper local_pri = controller.GetPRI();
    if (!local_pri || local_pri.GetCar().memory_address != local_car.memory_address) {
        reason = "local controller PRI does not own GetLocalCar";
        return false;
    }
    ServerWrapper server = gameWrapper->GetGameEventAsServer();
    if (!server) {
        reason = "local server wrapper is null";
        return false;
    }
    auto pris = server.GetPRIs();
    std::size_t matches = 0;
    PriWrapper matched(0);
    for (int index = 0; index < pris.Count(); ++index) {
        PriWrapper candidate = pris.Get(index);
        if (!candidate) {
            continue;
        }
        CarWrapper candidate_car = candidate.GetCar();
        if (candidate.memory_address == local_pri.memory_address && candidate_car &&
            candidate_car.memory_address == local_car.memory_address) {
            ++matches;
            matched = candidate;
        }
    }
    if (matches != 1) {
        reason = "expected exactly one server PRI matching controller and car; found " +
                 std::to_string(matches);
        return false;
    }
    if (local_car_address != nullptr) {
        *local_car_address = local_car.memory_address;
    }
    if (metadata != nullptr) {
        metadata->local_player_id = safe_pri_id(matched);
        metadata->local_player_name = matched.GetPlayerName().ToString();
        metadata->local_player_numeric_id = matched.GetPlayerID();
        metadata->local_team = static_cast<std::int8_t>(matched.GetTeamNum());
    }
    return true;
}

rivalrec::Frame RivalDemoRecorder::capture_frame(const ControllerInput& input,
                                                 std::uintptr_t local_car_address) {
    rivalrec::Frame frame;
    frame.sequence = next_sequence_++;
    EngineTAWrapper engine = gameWrapper->GetEngine();
    ServerWrapper server = gameWrapper->GetGameEventAsServer();
    frame.physics_frame = engine ? engine.GetPhysicsFrame() : -1;
    frame.replicated_physics_frame = engine ? engine.GetReplicatedPhysicsFrame() : -1;
    frame.engine_physics_time = engine ? engine.GetPhysicsTime() : 0.0F;
    frame.game_time_seconds = server ? server.GetSecondsElapsed() : 0.0F;
    frame.monotonic_ns = monotonic_ns();
    frame.utc_ns = rivalrec::utc_now_ns();
    frame.delta_monotonic_ns = previous_monotonic_ns_ == 0
                                   ? 0
                                   : frame.monotonic_ns - previous_monotonic_ns_;
    frame.delta_engine_seconds = previous_physics_frame_ < 0
                                     ? 0.0F
                                     : frame.engine_physics_time - previous_engine_time_;
    if (previous_physics_frame_ >= 0) {
        if (frame.physics_frame == previous_physics_frame_) {
            frame.status_flags |= kFrameDuplicate;
            ++counters_.duplicate_physics_frames;
        } else if (frame.physics_frame < previous_physics_frame_) {
            frame.status_flags |= kFrameOutOfOrder;
            ++counters_.out_of_order_physics_frames;
        } else if (frame.physics_frame > previous_physics_frame_ + 1) {
            frame.status_flags |= kFrameGap;
            frame.missing_physics_frames = static_cast<std::uint32_t>(
                frame.physics_frame - previous_physics_frame_ - 1);
            counters_.missing_physics_frames += frame.missing_physics_frames;
        }
    }
    previous_physics_frame_ = frame.physics_frame;
    previous_engine_time_ = frame.engine_physics_time;
    previous_monotonic_ns_ = frame.monotonic_ns;
    frame.native_input = native_input(input);
    frame.rival_action = rival_action(input);
    if (server) {
        BallWrapper ball = server.GetBall();
        if (ball) {
            frame.ball = capture_ball(server);
        }
        auto pris = server.GetPRIs();
        frame.cars.reserve(static_cast<std::size_t>(std::max(0, pris.Count())));
        for (int index = 0; index < pris.Count(); ++index) {
            PriWrapper pri = pris.Get(index);
            if (pri && !pri.IsSpectator()) {
                frame.cars.push_back(capture_car(pri, local_car_address));
            }
        }
        frame.match = capture_match(server);
    }
    frame.boost_pads = snapshot_pads(frame.game_time_seconds);
    return frame;
}

rivalrec::BallState RivalDemoRecorder::capture_ball(ServerWrapper server) const {
    BallWrapper ball = server.GetBall();
    rivalrec::BallState result;
    result.position = vector(ball.GetLocation());
    result.rotation = rotation(ball.GetRotation());
    result.linear_velocity = vector(ball.GetVelocity());
    result.angular_velocity = vector(ball.GetAngularVelocity());
    result.availability |= kBallTransform | kBallDynamics;
    result.gravity_z = ball.GetGravityZ();
    result.gravity_scale = ball.GetReplicatedBallGravityScale();
    result.availability |= kBallGravity;
    result.last_touch_time = ball.GetLastTouchTime();
    result.last_hit_world_time = ball.GetLastHitWorldTime();
    result.hit_team = static_cast<std::int8_t>(ball.GetHitTeamNum());
    result.availability |= kBallTouch;
    CarWrapper affector = ball.GetCurrentAffector();
    if (affector) {
        result.current_affector_id = safe_pri_id(affector.GetPRI());
        result.availability |= kBallAffector;
    }
    return result;
}

rivalrec::CarState RivalDemoRecorder::capture_car(PriWrapper pri,
                                                  std::uintptr_t local_car_address) const {
    rivalrec::CarState result;
    result.stable_id = safe_pri_id(pri);
    result.player_name = pri.GetPlayerName().ToString();
    result.player_id = pri.GetPlayerID();
    result.team = static_cast<std::int8_t>(pri.GetTeamNum());
    result.flags.is_bot = pri.GetbBot() != 0U;
    result.availability |= kCarIdentity;
    CarWrapper car = pri.GetCar();
    result.flags.car_present = static_cast<bool>(car);
    result.respawn_time_remaining = pri.GetRespawnTimeRemaining();
    result.flags.demolished = !car && result.respawn_time_remaining > 0;
    result.flags.is_local_human = car && car.memory_address == local_car_address &&
                                  !result.flags.is_bot;
    result.availability |= kCarLifecycle;
    if (!car) {
        return result;
    }
    result.position = vector(car.GetLocation());
    result.rotation = rotation(car.GetRotation());
    result.availability |= kCarTransform;
    result.linear_velocity = vector(car.GetVelocity());
    result.angular_velocity = vector(car.GetAngularVelocity());
    result.availability |= kCarDynamics;
    BoostWrapper boost = car.GetBoostComponent();
    if (boost) {
        result.boost = boost.GetCurrentBoostAmount();
        result.boost_component = component(boost);
        result.availability |= kCarBoost;
    }
    result.flags.on_ground = car.GetbOnGround() != 0U;
    result.flags.supersonic = car.GetbSuperSonic() != 0U;
    result.num_wheel_world_contacts = static_cast<std::int16_t>(car.GetNumWheelWorldContacts());
    result.num_wheel_contacts = static_cast<std::int16_t>(car.GetNumWheelContacts());
    result.time_off_ground = car.GetTimeOffGround();
    result.time_on_ground = car.GetTimeOnGround();
    result.availability |= kCarContactFlags;
    result.flags.jumped = car.GetbJumped() != 0U;
    result.flags.double_jumped = car.GetbDoubleJumped() != 0U;
    result.flags.can_jump = car.GetbCanJump() != 0U;
    result.flags.has_flip = car.HasFlip() != 0U;
    result.last_ball_touch_frame = car.GetLastBallTouchFrame();
    result.last_ball_impact_frame = car.GetLastBallImpactFrame();
    result.availability |= kCarJumpFlags;

    JumpComponentWrapper jump = car.GetJumpComponent();
    DoubleJumpComponentWrapper double_jump = car.GetDoubleJumpComponent();
    DodgeComponentWrapper dodge = car.GetDodgeComponent();
    FlipCarComponentWrapper flip = car.GetFlipComponent();
    result.jump_component = component(jump);
    result.double_jump_component = component(double_jump);
    result.dodge_component = component(dodge);
    if (dodge) {
        result.dodge_direction = vector(dodge.GetDodgeDirection());
    }
    result.flip_component = component(flip);
    if (flip) {
        result.flip_time = flip.GetFlipCarTime();
        result.flip_right = flip.GetbFlipRight() != 0U;
    }
    result.availability |= kCarComponents;

    result.native_input = native_input(car.GetInput());
    result.native_input_available = true;
    result.availability |= kCarInput;

    VehicleSimWrapper simulation = car.GetVehicleSim();
    if (simulation) {
        auto wheels = simulation.GetWheels();
        result.wheels.reserve(static_cast<std::size_t>(std::max(0, wheels.Count())));
        for (int index = 0; index < wheels.Count(); ++index) {
            WheelWrapper wheel = wheels.Get(index);
            if (!wheel) {
                continue;
            }
            const WheelContactData contact = wheel.GetContact();
            result.wheels.push_back(
                {.index = static_cast<std::int8_t>(wheel.GetWheelIndex()),
                 .has_contact = contact.bHasContact != 0U,
                 .has_world_contact = contact.bHasContactWithWorldGeometry != 0U,
                 .contact_change_time = contact.HasContactChangeTime,
                 .contact_location = vector(contact.Location),
                 .contact_normal = vector(contact.Normal),
                 .lateral_direction = vector(contact.LatDirection),
                 .longitudinal_direction = vector(contact.LongDirection),
                 .reference_location = vector(wheel.GetRefWheelLocation()),
                 .suspension_distance = wheel.GetSuspensionDistance(),
                 .spin_speed = wheel.GetSpinSpeed()});
        }
        result.availability |= kCarWheels;
    }
    return result;
}

rivalrec::MatchState RivalDemoRecorder::capture_match(ServerWrapper server) const {
    rivalrec::MatchState result;
    result.game_mode = server.GetMatchTypeName();
    result.map = gameWrapper->GetCurrentMap();
    result.match_guid = server.GetMatchGUID();
    result.availability |= kMatchIdentity;
    result.seconds_elapsed = server.GetSecondsElapsed();
    result.seconds_remaining = static_cast<float>(server.GetSecondsRemaining());
    result.total_game_time_played = server.GetTotalGameTimePlayed();
    result.overtime_time_played = server.GetOvertimeTimePlayed();
    result.round_number = server.GetRoundNum();
    result.countdown_number = server.GetReplicatedRoundCountDownNumber();
    result.seconds_remaining_countdown = server.GetSecondsRemainingCountdown();
    result.availability |= kMatchClock;
    result.flags.paused = gameWrapper->IsPaused();
    result.flags.overtime = server.GetbOverTime() != 0U;
    result.flags.round_active = server.GetbRoundActive() != 0U;
    result.flags.match_ended = server.GetbMatchEnded() != 0U;
    result.flags.ball_has_been_hit = server.GetbBallHasBeenHit() != 0U;
    result.flags.kickoff_or_countdown = !result.flags.round_active ||
                                        result.seconds_remaining_countdown > 0 ||
                                        !result.flags.ball_has_been_hit;
    result.availability |= kMatchFlags;
    auto teams = server.GetTeams();
    for (int index = 0; index < teams.Count(); ++index) {
        TeamWrapper team = teams.Get(index);
        if (!team) {
            continue;
        }
        const int team_index = team.GetTeamIndex();
        if (team_index == 0) {
            result.score_team_0 = team.GetScore();
        } else if (team_index == 1) {
            result.score_team_1 = team.GetScore();
        }
    }
    result.availability |= kMatchScore;
    return result;
}

std::vector<rivalrec::BoostPadState> RivalDemoRecorder::snapshot_pads(float game_time) {
    std::vector<rivalrec::BoostPadState> result;
    std::lock_guard lock(pads_mutex_);
    result.reserve(pads_.size());
    for (auto& [_, observation] : pads_) {
        rivalrec::BoostPadState state = observation.state;
        if (!state.active && state.respawn_delay > 0.0F) {
            const float elapsed = std::max(0.0F, game_time - observation.picked_up_game_time);
            state.cooldown_remaining = std::max(0.0F, state.respawn_delay - elapsed);
        }
        result.push_back(std::move(state));
    }
    return result;
}

bool RivalDemoRecorder::enqueue(WriteItem item, bool is_frame) {
    std::uint32_t limit = 4096;
    {
        std::lock_guard lock(session_mutex_);
        if (session_) {
            limit = session_->max_queue_frames;
        }
    }
    {
        std::lock_guard lock(queue_mutex_);
        if (queue_.size() >= limit) {
            if (is_frame) {
                ++counters_.queue_dropped_frames;
            }
            return false;
        }
        queue_.push_back(std::move(item));
    }
    queue_cv_.notify_one();
    return true;
}

void RivalDemoRecorder::writer_main(SessionConfig config) {
    std::vector<rivalrec::ChunkInfo> chunks;
    std::vector<rivalrec::FileInfo> journals;
    std::unique_ptr<rivalrec::ChunkWriter> chunk;
    std::ofstream events;
    std::ofstream markers;
    const std::filesystem::path events_path = config.directory / "events.jsonl";
    const std::filesystem::path markers_path = config.directory / "markers.jsonl";
    std::string failure;
    try {
        events.open(events_path, std::ios::binary | std::ios::trunc);
        markers.open(markers_path, std::ios::binary | std::ios::trunc);
        if (!events || !markers) {
            throw std::runtime_error("unable to open event or marker journal");
        }
        write_manifest(config, false, "in_progress", chunks, journals, "");
        std::uint32_t chunk_index = 0;
        while (true) {
            WriteItem item;
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return stop_requested_.load() || !queue_.empty(); });
                if (queue_.empty()) {
                    if (stop_requested_.load()) {
                        break;
                    }
                    continue;
                }
                item = std::move(queue_.front());
                queue_.pop_front();
            }
            if (auto* frame = std::get_if<rivalrec::Frame>(&item)) {
                if (!chunk) {
                    chunk = std::make_unique<rivalrec::ChunkWriter>(
                        config.directory, chunk_index, config.uuid, frame->sequence,
                        config.flush_frames);
                }
                chunk->append(*frame);
                ++counters_.written_frames;
                if (chunk->full(config.chunk_frames)) {
                    chunks.push_back(chunk->close());
                    chunk.reset();
                    ++chunk_index;
                    write_manifest(config, false, "in_progress", chunks, journals, "");
                }
            } else {
                const JournalRecord& record = std::get<JournalRecord>(item);
                std::ofstream& stream = record.journal == Journal::events ? events : markers;
                stream << record.json << '\n';
                stream.flush();
                if (!stream) {
                    throw std::runtime_error("failed while writing journal");
                }
                rivalrec::durable_flush_file(record.journal == Journal::events ? events_path
                                                                                : markers_path);
            }
        }
        if (chunk && chunk->frame_count() > 0) {
            chunks.push_back(chunk->close());
            chunk.reset();
        }
        events.flush();
        markers.flush();
        rivalrec::durable_flush_file(events_path);
        rivalrec::durable_flush_file(markers_path);
        events.close();
        markers.close();
        journals.push_back(rivalrec::inspect_file(config.directory, events_path));
        journals.push_back(rivalrec::inspect_file(config.directory, markers_path));
    } catch (const std::exception& exception) {
        failure = exception.what();
        writer_failed_ = true;
        recording_ = false;
        stop_requested_ = true;
        if (chunk) {
            chunk->abandon();
        }
        events.close();
        markers.close();
    }

    const std::string capture_end = rivalrec::utc_now_iso8601();
    const bool clean = clean_stop_ && failure.empty();
    std::string reason = failure.empty() ? stop_reason_ : "writer_exception:" + failure;
    try {
        write_manifest(config, clean, reason, chunks, journals, capture_end);
    } catch (...) {
        writer_failed_ = true;
    }
}

void RivalDemoRecorder::stop_session(bool clean, const std::string& reason) {
    const bool was_recording = recording_.exchange(false);
    if (!was_recording && !writer_thread_.joinable()) {
        return;
    }
    clean_stop_ = clean;
    stop_reason_ = reason;
    stop_requested_ = true;
    queue_cv_.notify_all();
    if (writer_thread_.joinable() && writer_thread_.get_id() != std::this_thread::get_id()) {
        writer_thread_.join();
    }
    std::optional<SessionConfig> config;
    {
        std::lock_guard lock(session_mutex_);
        config = session_;
        session_.reset();
        local_car_address_ = 0;
    }
    if (was_recording) {
        const std::string result = writer_failed_.load() ? "incomplete (writer failure)" :
                                                           (clean ? "clean" : "incomplete");
        log("recording stopped " + result + "; frames=" +
            std::to_string(counters_.written_frames.load()) +
            (config ? "; directory=" + config->directory.string() : ""));
    }
}

void RivalDemoRecorder::write_manifest(
    const SessionConfig& config, bool clean, const std::string& reason,
    const std::vector<rivalrec::ChunkInfo>& chunks,
    const std::vector<rivalrec::FileInfo>& journals,
    const std::string& capture_end_utc) const {
    const double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                           session_steady_start_)
                                .count();
    const std::uint64_t frames = counters_.written_frames.load();
    std::ostringstream json;
    json << std::setprecision(17);
    json << "{\n"
         << "  \"schema_name\": \"" << rivalrec::kSchemaName << "\",\n"
         << "  \"schema_version\": " << rivalrec::kSchemaVersion << ",\n"
         << "  \"session_uuid\": \"" << config.uuid.text << "\",\n"
         << "  \"session_type\": \"" << session_type_name(config.type) << "\",\n"
         << "  \"label\": \"" << rivalrec::json_escape(config.label) << "\",\n"
         << "  \"opponent_label\": \"" << rivalrec::json_escape(config.opponent_label)
         << "\",\n"
         << "  \"mechanic_label\": \"" << rivalrec::json_escape(config.mechanic_label)
         << "\",\n"
         << "  \"capture_start_utc\": \"" << config.capture_start_utc << "\",\n"
         << "  \"capture_start_local\": \"" << config.capture_start_local << "\",\n"
         << "  \"capture_end_utc\": \"" << capture_end_utc << "\",\n"
         << "  \"clean_termination\": " << bool_json(clean) << ",\n"
         << "  \"termination_reason\": \"" << rivalrec::json_escape(reason) << "\",\n"
         << "  \"rocket_league_build\": \""
         << rivalrec::json_escape(config.rocket_league_build) << "\",\n"
         << "  \"map\": \"" << rivalrec::json_escape(config.map) << "\",\n"
         << "  \"local_player\": {\"stable_id\": \""
         << rivalrec::json_escape(config.local_player_id) << "\", \"name\": \""
         << rivalrec::json_escape(config.local_player_name) << "\", \"player_id\": "
         << config.local_player_numeric_id << ", \"team\": "
         << static_cast<int>(config.local_team) << "},\n"
         << "  \"recorder_git_sha\": \"" RIVALREC_GIT_SHA "\",\n"
         << "  \"plugin_build\": \"" RIVALREC_PLUGIN_BUILD "\",\n"
         << "  \"bakkesmod_sdk_revision\": \"" RIVALREC_SDK_REVISION "\",\n"
         << "  \"bakkesmod_version\": " << config.bakkesmod_version << ",\n"
         << "  \"capture_hook\": \"Function TAGame.Car_TA.SetVehicleInput\",\n"
         << "  \"engine_physics_framerate_hz\": "
         << config.engine_physics_framerate << ",\n"
         << "  \"capture_policy\": \"read_only_native_input_application\",\n"
         << "  \"intended_contexts\": [\"freeplay\", \"offline_local\", \"rlbot_local\"],\n"
         << "  \"chunk_frames\": " << config.chunk_frames << ",\n"
         << "  \"flush_frames\": " << config.flush_frames << ",\n"
         << "  \"duration_seconds\": " << duration << ",\n"
         << "  \"final_frame_count\": " << frames << ",\n"
         << "  \"observed_capture_rate_hz\": "
         << (duration > 0.0 ? static_cast<double>(frames) / duration : 0.0) << ",\n"
         << "  \"attempted_frame_count\": " << counters_.attempted_frames.load() << ",\n"
         << "  \"enqueued_frame_count\": " << counters_.enqueued_frames.load() << ",\n"
         << "  \"queue_dropped_frame_count\": "
         << counters_.queue_dropped_frames.load() << ",\n"
         << "  \"duplicate_physics_frame_count\": "
         << counters_.duplicate_physics_frames.load() << ",\n"
         << "  \"out_of_order_physics_frame_count\": "
         << counters_.out_of_order_physics_frames.load() << ",\n"
         << "  \"missing_physics_frame_count\": "
         << counters_.missing_physics_frames.load() << ",\n"
         << "  \"identity_failure_count\": " << counters_.identity_failures.load() << ",\n"
         << "  \"event_record_count\": " << counters_.event_records.load() << ",\n"
         << "  \"marker_record_count\": " << counters_.marker_records.load() << ",\n"
         << "  \"chunks\": [";
    for (std::size_t index = 0; index < chunks.size(); ++index) {
        const auto& chunk = chunks[index];
        json << (index == 0 ? "\n" : ",\n")
             << "    {\"index\": " << chunk.index << ", \"path\": \""
             << rivalrec::json_escape(chunk.relative_path) << "\", \"frame_count\": "
             << chunk.frame_count << ", \"first_sequence\": " << chunk.first_sequence
             << ", \"last_sequence\": " << chunk.last_sequence << ", \"bytes\": "
             << chunk.bytes << ", \"sha256\": \"" << chunk.sha256
             << "\", \"complete\": " << bool_json(chunk.complete) << "}";
    }
    json << (chunks.empty() ? "],\n" : "\n  ],\n") << "  \"files\": [";
    for (std::size_t index = 0; index < journals.size(); ++index) {
        const auto& file = journals[index];
        json << (index == 0 ? "\n" : ",\n") << "    {\"path\": \""
             << rivalrec::json_escape(file.relative_path) << "\", \"bytes\": "
             << file.bytes << ", \"sha256\": \"" << file.sha256 << "\"}";
    }
    json << (journals.empty() ? "]\n" : "\n  ]\n") << "}\n";
    rivalrec::atomic_write_text(config.directory / "manifest.json", json.str());
}

void RivalDemoRecorder::record_event(const std::string& kind, const std::string& unreal_event,
                                     std::uintptr_t caller, const std::string& actor_id,
                                     const std::string& extra_json) {
    if (!recording_.load()) {
        return;
    }
    std::ostringstream json;
    json << journal_prefix() << ",\"kind\":\"" << rivalrec::json_escape(kind)
         << "\",\"unreal_event\":\"" << rivalrec::json_escape(unreal_event)
         << "\",\"caller\":\"" << pointer_id(caller) << "\",\"actor_id\":\""
         << rivalrec::json_escape(actor_id) << "\"";
    if (!extra_json.empty()) {
        json << ',' << extra_json;
    }
    json << '}';
    if (enqueue(JournalRecord{Journal::events, json.str()}, false)) {
        ++counters_.event_records;
    }
}

void RivalDemoRecorder::record_marker(const std::string& kind, const std::string& text) {
    if (!recording_.load()) {
        error("no active recording");
        return;
    }
    std::ostringstream json;
    json << journal_prefix() << ",\"kind\":\"" << rivalrec::json_escape(kind)
         << "\",\"text\":\"" << rivalrec::json_escape(text) << "\"}";
    if (enqueue(JournalRecord{Journal::markers, json.str()}, false)) {
        ++counters_.marker_records;
        log(kind + " recorded");
    } else {
        error("marker queue is full; marker was not recorded");
    }
}

std::string RivalDemoRecorder::journal_prefix() const {
    EngineTAWrapper engine = gameWrapper->GetEngine();
    ServerWrapper server = gameWrapper->GetGameEventAsServer();
    std::ostringstream json;
    json << "{\"sequence_boundary\":" << next_sequence_ << ",\"physics_frame\":"
         << (engine ? engine.GetPhysicsFrame() : -1) << ",\"engine_physics_time\":"
         << (engine ? engine.GetPhysicsTime() : 0.0F) << ",\"game_time_seconds\":"
         << (server ? server.GetSecondsElapsed() : 0.0F) << ",\"monotonic_ns\":"
         << monotonic_ns() << ",\"utc_ns\":" << rivalrec::utc_now_ns();
    return json.str();
}

std::string RivalDemoRecorder::configured_data_root() const {
    const std::string configured = cvarManager->getCvar("rivalrec_data_dir").getStringValue();
    if (!configured.empty()) {
        return std::filesystem::absolute(configured).string();
    }
    return (gameWrapper->GetDataFolder() / "rival2" / "human_demos").string();
}

std::string RivalDemoRecorder::join_arguments(const std::vector<std::string>& args,
                                              std::size_t start) const {
    std::ostringstream output;
    for (std::size_t index = start; index < args.size(); ++index) {
        if (index > start) {
            output << ' ';
        }
        output << args[index];
    }
    return output.str();
}

void RivalDemoRecorder::log(const std::string& message) const {
    cvarManager->log("[RivalRecorder] " + message);
}

void RivalDemoRecorder::error(const std::string& message) const {
    cvarManager->log("[RivalRecorder] ERROR: " + message);
    gameWrapper->Toast("Rival Recorder", message, "default", 5.0F, ToastType_Error);
}
