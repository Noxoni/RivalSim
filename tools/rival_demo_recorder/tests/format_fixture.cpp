#include "rivalrec/recording_format.hpp"

#include <filesystem>
#include <iostream>

namespace {

rivalrec::NativeInput input() {
    return {.throttle = 0.75F,
            .steer = -0.25F,
            .pitch = 0.5F,
            .yaw = -0.125F,
            .roll = 0.375F,
            .dodge_forward = -0.625F,
            .dodge_strafe = 0.875F,
            .handbrake = true,
            .jump = true,
            .activate_boost = false,
            .holding_boost = true,
            .jumped = true};
}

rivalrec::Frame frame() {
    rivalrec::Frame value;
    value.sequence = 7;
    value.physics_frame = 1007;
    value.replicated_physics_frame = 907;
    value.engine_physics_time = 10.25F;
    value.game_time_seconds = 20.25F;
    value.monotonic_ns = 1'058'333'331ULL;
    value.utc_ns = 2'058'333'331ULL;
    value.delta_monotonic_ns = 8'333'333ULL;
    value.delta_engine_seconds = 1.0F / 120.0F;
    value.native_input = input();
    value.rival_action = {.throttle = 0.75F,
                          .steer = -0.25F,
                          .pitch = 0.5F,
                          .yaw = -0.125F,
                          .roll = 0.375F,
                          .jump = true,
                          .boost = true,
                          .handbrake = true};
    value.ball = rivalrec::BallState{.availability = 0xFFU,
                                     .position = {-1.0F, 2.0F, 93.15F},
                                     .rotation = {11, 22, 33},
                                     .linear_velocity = {1000.0F, -500.0F, 250.0F},
                                     .angular_velocity = {2.0F, 3.0F, 4.0F},
                                     .gravity_z = -650.0F,
                                     .gravity_scale = 1.0F,
                                     .last_touch_time = 19.5F,
                                     .last_hit_world_time = 19.25F,
                                     .hit_team = 1,
                                     .current_affector_id = "Epic|human|0"};
    rivalrec::CarState car;
    car.stable_id = "Epic|human|0";
    car.player_name = "Human";
    car.player_id = 7;
    car.team = 1;
    car.flags = {.is_local_human = true,
                 .car_present = true,
                 .on_ground = false,
                 .supersonic = true,
                 .jumped = true,
                 .can_jump = false,
                 .has_flip = true};
    car.availability = 0x3FFU;
    car.position = {1.25F, -2.5F, 3.75F};
    car.rotation = {123, -456, 789};
    car.linear_velocity = {-100.5F, 200.25F, 300.125F};
    car.angular_velocity = {1.5F, -2.25F, 3.125F};
    car.boost = 42.5F;
    car.time_off_ground = 0.75F;
    car.last_ball_touch_frame = 100;
    car.last_ball_impact_frame = 101;
    car.respawn_time_remaining = 0;
    car.num_wheel_contacts = 1;
    car.boost_component = {true, 0.1F};
    car.jump_component = {false, 0.2F};
    car.dodge_component = {true, 0.15F};
    car.dodge_direction = {0.25F, -0.75F, 0.0F};
    car.native_input_available = true;
    car.native_input = input();
    car.wheels.push_back({.index = 0,
                          .has_contact = true,
                          .contact_change_time = 12.5F,
                          .contact_location = {10.0F, 11.0F, 12.0F},
                          .contact_normal = {0.0F, 0.0F, 1.0F},
                          .lateral_direction = {0.0F, 1.0F, 0.0F},
                          .longitudinal_direction = {1.0F, 0.0F, 0.0F},
                          .reference_location = {9.0F, 8.0F, 7.0F},
                          .suspension_distance = 6.5F,
                          .spin_speed = -5.25F});
    value.cars.push_back(car);
    value.match.game_mode = "Soccar";
    value.match.map = "Stadium_P";
    value.match.match_guid = "fixture-match-guid";
    value.match.seconds_elapsed = 20.0F;
    value.match.seconds_remaining = 280.0F;
    value.match.total_game_time_played = 20.0F;
    value.match.score_team_0 = 1;
    value.match.score_team_1 = 2;
    value.match.round_number = 3;
    value.match.flags.round_active = true;
    value.match.flags.ball_has_been_hit = true;
    value.match.availability = 0x3FU;
    value.boost_pads.push_back({.stable_id = "pickup:1234",
                                .position = {0.0F, -4240.0F, 70.0F},
                                .is_full_boost = true,
                                .active = true,
                                .cooldown_quality = 2,
                                .boost_type = 1,
                                .boost_amount = 1.0F,
                                .respawn_delay = 10.0F});
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: rivalrec_format_fixture <output-file>\n";
        return 2;
    }
    try {
        const std::vector<std::uint8_t> encoded = rivalrec::encode_frame(frame());
        const std::filesystem::path output(argv[1]);
        std::ofstream stream(output, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(encoded.data()),
                     static_cast<std::streamsize>(encoded.size()));
        stream.close();
        if (!stream) {
            throw std::runtime_error("failed to write fixture");
        }
        std::cout << encoded.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
