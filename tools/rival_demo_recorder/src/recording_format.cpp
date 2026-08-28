#include "rivalrec/recording_format.hpp"

#include <Windows.h>
#include <bcrypt.h>

#include <bit>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>

namespace rivalrec {
namespace {

constexpr std::array<std::uint8_t, 8> kChunkMagic{'R', 'I', 'V', 'R', 'D', 'M', 'O', '1'};
constexpr std::array<std::uint8_t, 4> kRecordMagic{'R', 'V', 'R', 'C'};
constexpr std::uint8_t kFrameRecord = 1;
constexpr std::uint8_t kFooterRecord = 2;

class Encoder {
public:
    template <typename T>
    void integer(T value) {
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned encoded = static_cast<Unsigned>(value);
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            bytes.push_back(static_cast<std::uint8_t>(encoded & 0xFFU));
            encoded >>= 8U;
        }
    }

    void floating(float value) { integer(std::bit_cast<std::uint32_t>(value)); }
    void boolean(bool value) { integer<std::uint8_t>(value ? 1U : 0U); }

    void raw(const std::uint8_t* begin, std::size_t size) {
        bytes.insert(bytes.end(), begin, begin + size);
    }

    template <std::size_t Size>
    void raw(const std::array<std::uint8_t, Size>& value) {
        raw(value.data(), value.size());
    }

    void string(const std::string& value) {
        if (value.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("Rival recorder UTF-8 string exceeds uint16 limit");
        }
        integer<std::uint16_t>(static_cast<std::uint16_t>(value.size()));
        raw(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    }

    std::vector<std::uint8_t> bytes;
};

void encode_vec3(Encoder& out, const Vec3& value) {
    out.floating(value.x);
    out.floating(value.y);
    out.floating(value.z);
}

void encode_rotator(Encoder& out, const Rotator& value) {
    out.integer(value.pitch);
    out.integer(value.yaw);
    out.integer(value.roll);
}

void encode_input(Encoder& out, const NativeInput& value) {
    out.floating(value.throttle);
    out.floating(value.steer);
    out.floating(value.pitch);
    out.floating(value.yaw);
    out.floating(value.roll);
    out.floating(value.dodge_forward);
    out.floating(value.dodge_strafe);
    out.boolean(value.handbrake);
    out.boolean(value.jump);
    out.boolean(value.activate_boost);
    out.boolean(value.holding_boost);
    out.boolean(value.jumped);
    out.integer<std::uint8_t>(0);
    out.integer<std::uint8_t>(0);
    out.integer<std::uint8_t>(0);
}

void encode_action(Encoder& out, const RivalAction& value) {
    out.floating(value.throttle);
    out.floating(value.steer);
    out.floating(value.pitch);
    out.floating(value.yaw);
    out.floating(value.roll);
    out.boolean(value.jump);
    out.boolean(value.boost);
    out.boolean(value.handbrake);
    out.integer<std::uint8_t>(0);
}

void encode_component(Encoder& out, const ComponentState& value) {
    out.boolean(value.active);
    out.floating(value.activity_time);
}

std::uint32_t car_flags(const CarFlags& flags) {
    return (static_cast<std::uint32_t>(flags.is_local_human) << 0U) |
           (static_cast<std::uint32_t>(flags.is_bot) << 1U) |
           (static_cast<std::uint32_t>(flags.car_present) << 2U) |
           (static_cast<std::uint32_t>(flags.demolished) << 3U) |
           (static_cast<std::uint32_t>(flags.on_ground) << 4U) |
           (static_cast<std::uint32_t>(flags.supersonic) << 5U) |
           (static_cast<std::uint32_t>(flags.jumped) << 6U) |
           (static_cast<std::uint32_t>(flags.double_jumped) << 7U) |
           (static_cast<std::uint32_t>(flags.can_jump) << 8U) |
           (static_cast<std::uint32_t>(flags.has_flip) << 9U);
}

std::uint32_t match_flags(const MatchFlags& flags) {
    return (static_cast<std::uint32_t>(flags.paused) << 0U) |
           (static_cast<std::uint32_t>(flags.overtime) << 1U) |
           (static_cast<std::uint32_t>(flags.round_active) << 2U) |
           (static_cast<std::uint32_t>(flags.match_ended) << 3U) |
           (static_cast<std::uint32_t>(flags.ball_has_been_hit) << 4U) |
           (static_cast<std::uint32_t>(flags.kickoff_or_countdown) << 5U);
}

std::string uppercase_hex(const std::uint8_t* bytes, std::size_t count) {
    std::ostringstream output;
    output << std::uppercase << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < count; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    return output.str();
}

void check_nt(NTSTATUS status, const char* operation) {
    if (status < 0) {
        throw std::runtime_error(std::string(operation) + " failed with NTSTATUS " +
                                 std::to_string(status));
    }
}

}  // namespace

std::vector<std::uint8_t> encode_frame(const Frame& frame) {
    Encoder out;
    out.integer(frame.sequence);
    out.integer(frame.physics_frame);
    out.integer(frame.replicated_physics_frame);
    out.floating(frame.engine_physics_time);
    out.floating(frame.game_time_seconds);
    out.integer(frame.monotonic_ns);
    out.integer(frame.utc_ns);
    out.integer(frame.delta_monotonic_ns);
    out.floating(frame.delta_engine_seconds);
    out.integer(frame.status_flags);
    out.integer(frame.missing_physics_frames);
    encode_input(out, frame.native_input);
    encode_action(out, frame.rival_action);

    out.boolean(frame.ball.has_value());
    if (frame.ball.has_value()) {
        const BallState& ball = *frame.ball;
        out.integer(ball.availability);
        encode_vec3(out, ball.position);
        encode_rotator(out, ball.rotation);
        encode_vec3(out, ball.linear_velocity);
        encode_vec3(out, ball.angular_velocity);
        out.floating(ball.gravity_z);
        out.floating(ball.gravity_scale);
        out.floating(ball.last_touch_time);
        out.floating(ball.last_hit_world_time);
        out.integer(ball.hit_team);
        out.string(ball.current_affector_id);
    }

    if (frame.cars.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("Rival recorder car count exceeds uint16");
    }
    out.integer<std::uint16_t>(static_cast<std::uint16_t>(frame.cars.size()));
    for (const CarState& car : frame.cars) {
        out.string(car.stable_id);
        out.string(car.player_name);
        out.integer(car.player_id);
        out.integer(car.team);
        out.integer(car_flags(car.flags));
        out.integer(car.availability);
        encode_vec3(out, car.position);
        encode_rotator(out, car.rotation);
        encode_vec3(out, car.linear_velocity);
        encode_vec3(out, car.angular_velocity);
        out.floating(car.boost);
        out.floating(car.time_off_ground);
        out.floating(car.time_on_ground);
        out.integer(car.last_ball_touch_frame);
        out.integer(car.last_ball_impact_frame);
        out.integer(car.respawn_time_remaining);
        out.integer(car.num_wheel_world_contacts);
        out.integer(car.num_wheel_contacts);
        encode_component(out, car.boost_component);
        encode_component(out, car.jump_component);
        encode_component(out, car.double_jump_component);
        encode_component(out, car.dodge_component);
        encode_vec3(out, car.dodge_direction);
        encode_component(out, car.flip_component);
        out.floating(car.flip_time);
        out.boolean(car.flip_right);
        out.boolean(car.native_input_available);
        encode_input(out, car.native_input);
        if (car.wheels.size() > std::numeric_limits<std::uint8_t>::max()) {
            throw std::runtime_error("Rival recorder wheel count exceeds uint8");
        }
        out.integer<std::uint8_t>(static_cast<std::uint8_t>(car.wheels.size()));
        for (const WheelState& wheel : car.wheels) {
            out.integer(wheel.index);
            const std::uint8_t flags = static_cast<std::uint8_t>(wheel.has_contact) |
                                       (static_cast<std::uint8_t>(wheel.has_world_contact) << 1U);
            out.integer(flags);
            out.floating(wheel.contact_change_time);
            encode_vec3(out, wheel.contact_location);
            encode_vec3(out, wheel.contact_normal);
            encode_vec3(out, wheel.lateral_direction);
            encode_vec3(out, wheel.longitudinal_direction);
            encode_vec3(out, wheel.reference_location);
            out.floating(wheel.suspension_distance);
            out.floating(wheel.spin_speed);
        }
    }

    out.string(frame.match.game_mode);
    out.string(frame.match.map);
    out.string(frame.match.match_guid);
    out.floating(frame.match.seconds_elapsed);
    out.floating(frame.match.seconds_remaining);
    out.floating(frame.match.total_game_time_played);
    out.floating(frame.match.overtime_time_played);
    out.integer(frame.match.score_team_0);
    out.integer(frame.match.score_team_1);
    out.integer(frame.match.round_number);
    out.integer(frame.match.countdown_number);
    out.integer(frame.match.seconds_remaining_countdown);
    out.integer(match_flags(frame.match.flags));
    out.integer(frame.match.availability);
    if (frame.boost_pads.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("Rival recorder boost-pad count exceeds uint16");
    }
    out.integer<std::uint16_t>(static_cast<std::uint16_t>(frame.boost_pads.size()));
    for (const BoostPadState& pad : frame.boost_pads) {
        out.string(pad.stable_id);
        encode_vec3(out, pad.position);
        out.boolean(pad.is_full_boost);
        const std::uint8_t flags = static_cast<std::uint8_t>(pad.active) |
                                   (static_cast<std::uint8_t>(pad.picked_up) << 1U);
        out.integer(flags);
        out.integer(pad.cooldown_quality);
        out.integer(pad.boost_type);
        out.floating(pad.boost_amount);
        out.floating(pad.respawn_delay);
        out.floating(pad.cooldown_remaining);
    }
    return out.bytes;
}

std::uint32_t crc32(const std::vector<std::uint8_t>& bytes) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (std::uint8_t value : bytes) {
        crc ^= value;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0U - (crc & 1U);
            crc = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

std::string sha256_file(const std::filesystem::path& path) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    check_nt(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0),
             "BCryptOpenAlgorithmProvider");
    try {
        DWORD object_size = 0;
        DWORD result_size = 0;
        check_nt(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                                   reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size),
                                   &result_size, 0),
                 "BCryptGetProperty(object length)");
        DWORD digest_size = 0;
        check_nt(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                                   reinterpret_cast<PUCHAR>(&digest_size), sizeof(digest_size),
                                   &result_size, 0),
                 "BCryptGetProperty(hash length)");
        std::vector<std::uint8_t> object(object_size);
        std::vector<std::uint8_t> digest(digest_size);
        check_nt(BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0),
                 "BCryptCreateHash");
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("unable to open chunk for SHA-256: " + path.string());
        }
        std::array<char, 1024 * 1024> buffer{};
        while (stream) {
            stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = stream.gcount();
            if (count > 0) {
                check_nt(BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()),
                                        static_cast<ULONG>(count), 0),
                         "BCryptHashData");
            }
        }
        check_nt(BCryptFinishHash(hash, digest.data(), digest_size, 0), "BCryptFinishHash");
        BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        return uppercase_hex(digest.data(), digest.size());
    } catch (...) {
        if (hash != nullptr) {
            BCryptDestroyHash(hash);
        }
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw;
    }
}

FileInfo inspect_file(const std::filesystem::path& session_directory,
                      const std::filesystem::path& path) {
    FileInfo info;
    info.relative_path = std::filesystem::relative(path, session_directory).generic_string();
    info.bytes = std::filesystem::file_size(path);
    info.sha256 = sha256_file(path);
    return info;
}

SessionUuid make_session_uuid() {
    SessionUuid result;
    check_nt(BCryptGenRandom(nullptr, result.bytes.data(), static_cast<ULONG>(result.bytes.size()),
                             BCRYPT_USE_SYSTEM_PREFERRED_RNG),
             "BCryptGenRandom");
    result.bytes[6] = static_cast<std::uint8_t>((result.bytes[6] & 0x0FU) | 0x40U);
    result.bytes[8] = static_cast<std::uint8_t>((result.bytes[8] & 0x3FU) | 0x80U);
    const std::string hex = uppercase_hex(result.bytes.data(), result.bytes.size());
    result.text = hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-" + hex.substr(12, 4) + "-" +
                  hex.substr(16, 4) + "-" + hex.substr(20, 12);
    return result;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec;
                } else {
                    output << character;
                }
        }
    }
    return output.str();
}

std::string utc_now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_s(&utc, &seconds);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3)
           << std::setfill('0') << milliseconds.count() << 'Z';
    return output.str();
}

std::uint64_t utc_now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count());
}

void atomic_write_text(const std::filesystem::path& path, const std::string& text) {
    const std::filesystem::path temporary = path.wstring() + L".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("unable to write temporary manifest: " + temporary.string());
        }
        stream.write(text.data(), static_cast<std::streamsize>(text.size()));
        stream.flush();
        if (!stream) {
            throw std::runtime_error("failed while writing temporary manifest");
        }
    }
    durable_flush_file(temporary);
    if (MoveFileExW(temporary.c_str(), path.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        throw std::runtime_error("unable to publish manifest atomically: Win32 error " +
                                 std::to_string(GetLastError()));
    }
}

void durable_flush_file(const std::filesystem::path& path) {
    const HANDLE handle = CreateFileW(
        path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("unable to open file for durable flush: " + path.string() +
                                 " (Win32 error " + std::to_string(GetLastError()) + ")");
    }
    const BOOL flushed = FlushFileBuffers(handle);
    const DWORD error = flushed == 0 ? GetLastError() : ERROR_SUCCESS;
    CloseHandle(handle);
    if (flushed == 0) {
        throw std::runtime_error("durable flush failed for " + path.string() +
                                 " (Win32 error " + std::to_string(error) + ")");
    }
}

ChunkWriter::ChunkWriter(const std::filesystem::path& session_directory,
                         std::uint32_t chunk_index, const SessionUuid& session_uuid,
                         std::uint64_t first_sequence, std::uint32_t flush_frames)
    : session_directory_(session_directory),
      chunk_index_(chunk_index),
      flush_frames_(flush_frames),
      first_sequence_(first_sequence),
      last_sequence_(first_sequence) {
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << chunk_index_ << ".rvr";
    final_path_ = session_directory_ / "chunks" / name.str();
    partial_path_ = final_path_.wstring() + L".partial";
    stream_.open(partial_path_, std::ios::binary | std::ios::trunc);
    if (!stream_) {
        throw std::runtime_error("unable to create chunk: " + partial_path_.string());
    }
    Encoder header;
    header.raw(kChunkMagic);
    header.integer(kSchemaVersion);
    header.integer(chunk_index_);
    header.raw(session_uuid.bytes);
    header.integer(first_sequence_);
    stream_.write(reinterpret_cast<const char*>(header.bytes.data()),
                  static_cast<std::streamsize>(header.bytes.size()));
}

ChunkWriter::~ChunkWriter() {
    if (!closed_ && stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
}

void ChunkWriter::write_record(std::uint8_t type, const std::vector<std::uint8_t>& payload) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Rival recorder record exceeds uint32 byte limit");
    }
    Encoder header;
    header.raw(kRecordMagic);
    header.integer(type);
    header.integer<std::uint8_t>(0);
    header.integer<std::uint8_t>(0);
    header.integer<std::uint8_t>(0);
    header.integer<std::uint32_t>(static_cast<std::uint32_t>(payload.size()));
    header.integer(crc32(payload));
    stream_.write(reinterpret_cast<const char*>(header.bytes.data()),
                  static_cast<std::streamsize>(header.bytes.size()));
    stream_.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
    if (!stream_) {
        throw std::runtime_error("failed while appending Rival recorder chunk");
    }
}

void ChunkWriter::append(const Frame& frame) {
    if (closed_) {
        throw std::logic_error("cannot append a closed chunk");
    }
    write_record(kFrameRecord, encode_frame(frame));
    ++frame_count_;
    last_sequence_ = frame.sequence;
    if (flush_frames_ > 0U && frame_count_ % flush_frames_ == 0U) {
        stream_.flush();
        durable_flush_file(partial_path_);
    }
}

bool ChunkWriter::full(std::uint32_t chunk_frames) const {
    return frame_count_ >= chunk_frames;
}

std::uint64_t ChunkWriter::frame_count() const { return frame_count_; }

ChunkInfo ChunkWriter::close() {
    if (closed_) {
        throw std::logic_error("chunk already closed");
    }
    Encoder footer;
    footer.integer(frame_count_);
    footer.integer(last_sequence_);
    write_record(kFooterRecord, footer.bytes);
    stream_.flush();
    durable_flush_file(partial_path_);
    stream_.close();
    std::filesystem::rename(partial_path_, final_path_);
    closed_ = true;
    ChunkInfo info;
    info.index = chunk_index_;
    info.relative_path = "chunks/" + final_path_.filename().string();
    info.frame_count = frame_count_;
    info.first_sequence = first_sequence_;
    info.last_sequence = last_sequence_;
    info.bytes = std::filesystem::file_size(final_path_);
    info.sha256 = sha256_file(final_path_);
    info.complete = true;
    return info;
}

void ChunkWriter::abandon() {
    if (stream_.is_open()) {
        stream_.flush();
        try {
            durable_flush_file(partial_path_);
        } catch (...) {
            // Preserve the original writer failure; the partial reader still verifies every record.
        }
        stream_.close();
    }
    closed_ = true;
}

}  // namespace rivalrec
