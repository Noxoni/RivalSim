#include "rivalrec/recorder_policy.hpp"

#include <cassert>
#include <cstdint>
#include <string>

namespace {

rivalrec::Frame frame(std::int32_t physics_frame, float throttle) {
    rivalrec::Frame result;
    result.physics_frame = physics_frame;
    result.engine_physics_time = static_cast<float>(physics_frame) / 120.0F;
    result.monotonic_ns = static_cast<std::uint64_t>(physics_frame) * 8'333'333ULL;
    result.native_input.throttle = throttle;
    result.rival_action.throttle = throttle;
    return result;
}

void test_local_car_rebinding() {
    constexpr std::uintptr_t first = 0x1000;
    constexpr std::uintptr_t second = 0x2000;
    constexpr std::uintptr_t third = 0x3000;
    const std::string stable_id = "Steam|stable-human|0";

    auto decision = rivalrec::evaluate_local_car_binding(
        stable_id, first, {true, stable_id, first}, first);
    assert(decision.identity_valid);
    assert(!decision.rebound);
    assert(decision.capture_caller);

    decision = rivalrec::evaluate_local_car_binding(
        stable_id, first, {true, stable_id, second}, second);
    assert(decision.identity_valid);
    assert(decision.rebound);
    assert(decision.previous_car_address == first);
    assert(decision.current_car_address == second);
    assert(decision.capture_caller);

    decision = rivalrec::evaluate_local_car_binding(
        stable_id, second, {true, stable_id, third}, third);
    assert(decision.identity_valid);
    assert(decision.rebound);
    assert(decision.previous_car_address == second);
    assert(decision.current_car_address == third);
    assert(decision.capture_caller);

    decision = rivalrec::evaluate_local_car_binding(
        stable_id, third, {true, stable_id, third}, 0x9999);
    assert(decision.identity_valid);
    assert(!decision.rebound);
    assert(!decision.capture_caller);
}

void test_identity_failure_is_not_guessed() {
    const std::string stable_id = "Epic|stable-human|0";
    auto decision = rivalrec::evaluate_local_car_binding(
        stable_id, 0x1000, {false, stable_id, 0x2000}, 0x2000);
    assert(!decision.identity_valid);
    decision = rivalrec::evaluate_local_car_binding(
        stable_id, 0x1000, {true, "Epic|different-human|0", 0x2000}, 0x2000);
    assert(!decision.identity_valid);
}

void test_sequence_continues_across_multiple_replacements() {
    const std::string stable_id = "Steam|stable-human|0";
    std::uintptr_t bound = 0x1000;
    rivalrec::TickFrameReducer reducer;
    (void)reducer.submit(frame(400, 0.1F));
    std::int32_t physics_frame = 401;

    for (const std::uintptr_t replacement : {0x2000U, 0x3000U, 0x4000U}) {
        const auto binding = rivalrec::evaluate_local_car_binding(
            stable_id, bound, {true, stable_id, replacement}, replacement);
        assert(binding.identity_valid);
        assert(binding.rebound);
        assert(binding.capture_caller);
        bound = replacement;
        const auto reduction = reducer.submit(frame(physics_frame++, 0.2F));
        assert(reduction.kind == rivalrec::TickSubmitKind::advanced);
        assert(reduction.frame_to_publish);
    }
    const auto final_frame = reducer.flush();
    assert(final_frame);
    assert(final_frame->sequence == 3);
}

void test_freeplay_reset_reuses_logical_player_identity() {
    const std::string stable_id = "Epic|freeplay-human|0";
    const auto binding = rivalrec::evaluate_local_car_binding(
        stable_id, 0xA000, {true, stable_id, 0xB000}, 0xB000);
    assert(binding.identity_valid);
    assert(binding.rebound);
    assert(binding.capture_caller);
}

void test_last_same_tick_callback_wins_without_duplicate_frame() {
    rivalrec::TickFrameReducer reducer;
    auto result = reducer.submit(frame(100, 0.25F));
    assert(result.kind == rivalrec::TickSubmitKind::buffered);
    assert(reducer.next_sequence_boundary() == 1);

    result = reducer.submit(frame(100, -0.75F));
    assert(result.kind == rivalrec::TickSubmitKind::duplicate_replaced);
    assert(result.suppressed_duplicate);
    assert(result.retained_duplicate);
    assert(result.suppressed_duplicate->native_input.throttle == 0.25F);
    assert(result.retained_duplicate->native_input.throttle == -0.75F);
    assert(reducer.next_sequence_boundary() == 1);

    result = reducer.submit(frame(101, 1.0F));
    assert(result.kind == rivalrec::TickSubmitKind::advanced);
    assert(result.frame_to_publish);
    assert(result.frame_to_publish->sequence == 0);
    assert(result.frame_to_publish->physics_frame == 100);
    assert(result.frame_to_publish->native_input.throttle == -0.75F);
    assert((result.frame_to_publish->status_flags &
            rivalrec::kFrameDuplicatePhysics) == 0U);

    const auto final_frame = reducer.flush();
    assert(final_frame);
    assert(final_frame->sequence == 1);
    assert(final_frame->physics_frame == 101);
    assert(final_frame->native_input.throttle == 1.0F);
}

void test_gap_and_out_of_order_are_explicit() {
    rivalrec::TickFrameReducer reducer;
    (void)reducer.submit(frame(200, 0.0F));
    auto result = reducer.submit(frame(203, 0.0F));
    assert(result.kind == rivalrec::TickSubmitKind::advanced);
    assert(result.missing_physics_frames == 2);
    const auto pending = reducer.flush();
    assert(pending);
    assert(pending->missing_physics_frames == 2);
    assert((pending->status_flags & rivalrec::kFramePhysicsGap) != 0U);

    reducer.reset();
    (void)reducer.submit(frame(300, 0.0F));
    result = reducer.submit(frame(299, 0.0F));
    assert(result.kind == rivalrec::TickSubmitKind::out_of_order_rejected);
    assert(reducer.next_sequence_boundary() == 1);
}

}  // namespace

int main() {
    test_local_car_rebinding();
    test_identity_failure_is_not_guessed();
    test_sequence_continues_across_multiple_replacements();
    test_freeplay_reset_reuses_logical_player_identity();
    test_last_same_tick_callback_wins_without_duplicate_frame();
    test_gap_and_out_of_order_are_explicit();
    return 0;
}
