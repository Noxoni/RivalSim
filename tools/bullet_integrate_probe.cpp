#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "LinearMath/btTransformUtil.h"

namespace {

void PrintFloat(const char* name, btScalar value) {
    std::printf(
        "%s=%.9g bits=%08X\n",
        name,
        static_cast<double>(value),
        std::bit_cast<std::uint32_t>(static_cast<float>(value)));
}

void PrintQuaternion(const char* name, const btQuaternion& value) {
    std::printf("%s=", name);
    for (int index = 0; index < 4; ++index) {
        const float component = static_cast<float>(value[index]);
        std::printf(
            "%s%.9g/%08X",
            index == 0 ? "" : ",",
            static_cast<double>(component),
            std::bit_cast<std::uint32_t>(component));
    }
    std::printf("\n");
}

void PrintMatrix(const char* name, const btMatrix3x3& value) {
    std::printf("%s=", name);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const float component = static_cast<float>(value[row][column]);
            std::printf(
                "%s%.9g/%08X",
                row == 0 && column == 0 ? "" : ",",
                static_cast<double>(component),
                std::bit_cast<std::uint32_t>(component));
        }
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 && argc != 8 && argc != 17) {
        std::fprintf(
            stderr,
            "usage: bullet_integrate_probe AX AY AZ DT "
            "[SPLIT_X SPLIT_Y SPLIT_Z]\n"
            "   or: bullet_integrate_probe M00..M22 AX AY AZ DT "
            "SPLIT_X SPLIT_Y SPLIT_Z\n");
        return 2;
    }
    const int angularOffset = argc == 17 ? 10 : 1;
    const btVector3 angular(
        static_cast<btScalar>(std::strtof(argv[angularOffset], nullptr)),
        static_cast<btScalar>(std::strtof(argv[angularOffset + 1], nullptr)),
        static_cast<btScalar>(std::strtof(argv[angularOffset + 2], nullptr)));
    const btScalar timeStep = static_cast<btScalar>(
        std::strtof(argv[angularOffset + 3], nullptr));
    const int splitOffset = angularOffset + 4;
    const btVector3 split = argc == 8 || argc == 17
        ? btVector3(
              static_cast<btScalar>(std::strtof(argv[splitOffset], nullptr)),
              static_cast<btScalar>(std::strtof(argv[splitOffset + 1], nullptr)),
              static_cast<btScalar>(std::strtof(argv[splitOffset + 2], nullptr)))
        : btVector3(btScalar(0.0), btScalar(0.0), btScalar(0.0));

    const btScalar angleSquared = angular.length2();
    btScalar angle = btScalar(0.0);
    if (angleSquared > SIMD_EPSILON) {
        angle = btSqrt(angleSquared);
    }
    if (angle * timeStep > ANGULAR_MOTION_THRESHOLD) {
        angle = ANGULAR_MOTION_THRESHOLD / timeStep;
    }
    btVector3 axis;
    if (angle < btScalar(0.001)) {
        axis = angular
            * (btScalar(0.5) * timeStep
               - (timeStep * timeStep * timeStep)
                   * btScalar(0.020833333333) * angle * angle);
    } else {
        axis = angular * (btSin(btScalar(0.5) * angle * timeStep) / angle);
    }
    const btScalar cosine = btCos(angle * timeStep * btScalar(0.5));
    btQuaternion predicted(axis.x(), axis.y(), axis.z(), cosine);
    predicted.safeNormalize();

    btTransform source;
    source.setIdentity();
    if (argc == 17) {
        source.setBasis(btMatrix3x3(
            static_cast<btScalar>(std::strtof(argv[1], nullptr)),
            static_cast<btScalar>(std::strtof(argv[2], nullptr)),
            static_cast<btScalar>(std::strtof(argv[3], nullptr)),
            static_cast<btScalar>(std::strtof(argv[4], nullptr)),
            static_cast<btScalar>(std::strtof(argv[5], nullptr)),
            static_cast<btScalar>(std::strtof(argv[6], nullptr)),
            static_cast<btScalar>(std::strtof(argv[7], nullptr)),
            static_cast<btScalar>(std::strtof(argv[8], nullptr)),
            static_cast<btScalar>(std::strtof(argv[9], nullptr))));
    }
    btTransform afterSplit;
    btTransformUtil::integrateTransform(
        source,
        btVector3(btScalar(0.0), btScalar(0.0), btScalar(0.0)),
        split * btScalar(0.1),
        timeStep,
        afterSplit);
    btTransform integrated;
    btTransformUtil::integrateTransform(
        afterSplit,
        btVector3(btScalar(0.0), btScalar(0.0), btScalar(0.0)),
        angular,
        timeStep,
        integrated);

    PrintFloat("angle_squared", angleSquared);
    PrintFloat("ball_damping_factor", btPow(btScalar(0.97), timeStep));
    PrintFloat("angle", angle);
    PrintFloat("half_step_angle", btScalar(0.5) * angle * timeStep);
    PrintFloat("sine", btSin(btScalar(0.5) * angle * timeStep));
    PrintFloat("axis_scale", btSin(btScalar(0.5) * angle * timeStep) / angle);
    PrintFloat("cosine", cosine);
    PrintQuaternion("predicted_quaternion", predicted);
    PrintMatrix("after_split_matrix", afterSplit.getBasis());
    PrintMatrix("predicted_matrix", integrated.getBasis());
    return 0;
}
