#pragma once

#include "rivalrec/recording_format.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace rivalrec {

inline constexpr std::uint32_t kFrameDuplicatePhysics = 1U << 0U;
inline constexpr std::uint32_t kFrameOutOfOrderPhysics = 1U << 1U;
inline constexpr std::uint32_t kFramePhysicsGap = 1U << 2U;

struct LocalHumanResolution {
    bool unique{};
    std::string stable_player_id;
    std::uintptr_t car_address{};
};

struct LocalCarBindingDecision {
    bool identity_valid{};
    bool rebound{};
    bool capture_caller{};
    std::uintptr_t previous_car_address{};
    std::uintptr_t current_car_address{};
    std::string failure_reason;
};

inline LocalCarBindingDecision evaluate_local_car_binding(
    const std::string& session_stable_player_id,
    std::uintptr_t bound_car_address,
    const LocalHumanResolution& resolution,
    std::uintptr_t callback_car_address) {
    LocalCarBindingDecision result;
    result.previous_car_address = bound_car_address;
    result.current_car_address = resolution.car_address;
    if (!resolution.unique || resolution.stable_player_id.empty() ||
        resolution.car_address == 0U) {
        result.failure_reason = "local human identity is not uniquely resolved";
        return result;
    }
    if (session_stable_player_id.empty() ||
        resolution.stable_player_id != session_stable_player_id) {
        result.failure_reason = "resolved stable player identity changed";
        return result;
    }
    if (bound_car_address == 0U) {
        result.failure_reason = "session has no bound local car";
        return result;
    }
    result.identity_valid = true;
    result.rebound = resolution.car_address != bound_car_address;
    result.capture_caller = callback_car_address == resolution.car_address;
    return result;
}

enum class TickSubmitKind {
    buffered,
    advanced,
    duplicate_replaced,
    out_of_order_rejected,
};

struct TickSubmitResult {
    TickSubmitKind kind{TickSubmitKind::buffered};
    std::optional<Frame> frame_to_publish;
    std::optional<Frame> suppressed_duplicate;
    std::optional<Frame> retained_duplicate;
    std::uint32_t missing_physics_frames{};
};

class TickFrameReducer {
public:
    void reset() {
        pending_.reset();
        last_finalized_.reset();
        next_sequence_ = 0;
    }

    [[nodiscard]] TickSubmitResult submit(Frame frame) {
        TickSubmitResult result;
        frame.status_flags = 0;
        frame.missing_physics_frames = 0;
        if (!pending_) {
            frame.sequence = next_sequence_++;
            normalize_timing(frame);
            pending_ = std::move(frame);
            return result;
        }

        if (frame.physics_frame < pending_->physics_frame) {
            result.kind = TickSubmitKind::out_of_order_rejected;
            return result;
        }
        if (frame.physics_frame == pending_->physics_frame) {
            result.kind = TickSubmitKind::duplicate_replaced;
            result.suppressed_duplicate = std::move(pending_);
            frame.sequence = result.suppressed_duplicate->sequence;
            normalize_timing(frame);
            pending_ = std::move(frame);
            result.retained_duplicate = *pending_;
            return result;
        }

        result.kind = TickSubmitKind::advanced;
        result.frame_to_publish = std::move(pending_);
        last_finalized_ = TimingIdentity::from(*result.frame_to_publish);
        frame.sequence = next_sequence_++;
        normalize_timing(frame);
        result.missing_physics_frames = frame.missing_physics_frames;
        pending_ = std::move(frame);
        return result;
    }

    [[nodiscard]] std::optional<Frame> flush() {
        if (!pending_) {
            return std::nullopt;
        }
        std::optional<Frame> result = std::move(pending_);
        pending_.reset();
        last_finalized_ = TimingIdentity::from(*result);
        return result;
    }

    [[nodiscard]] std::uint64_t next_sequence_boundary() const {
        return next_sequence_;
    }

private:
    struct TimingIdentity {
        std::int32_t physics_frame{};
        float engine_physics_time{};
        std::uint64_t monotonic_ns{};

        [[nodiscard]] static TimingIdentity from(const Frame& frame) {
            return {frame.physics_frame, frame.engine_physics_time, frame.monotonic_ns};
        }
    };

    void normalize_timing(Frame& frame) const {
        frame.status_flags &= ~(kFrameDuplicatePhysics | kFrameOutOfOrderPhysics |
                                kFramePhysicsGap);
        frame.missing_physics_frames = 0;
        if (!last_finalized_) {
            frame.delta_monotonic_ns = 0;
            frame.delta_engine_seconds = 0.0F;
            return;
        }
        frame.delta_monotonic_ns = frame.monotonic_ns >= last_finalized_->monotonic_ns
                                       ? frame.monotonic_ns - last_finalized_->monotonic_ns
                                       : 0;
        frame.delta_engine_seconds =
            frame.engine_physics_time - last_finalized_->engine_physics_time;
        if (frame.physics_frame > last_finalized_->physics_frame + 1) {
            frame.status_flags |= kFramePhysicsGap;
            frame.missing_physics_frames = static_cast<std::uint32_t>(
                frame.physics_frame - last_finalized_->physics_frame - 1);
        }
    }

    std::optional<Frame> pending_;
    std::optional<TimingIdentity> last_finalized_;
    std::uint64_t next_sequence_{};
};

}  // namespace rivalrec
