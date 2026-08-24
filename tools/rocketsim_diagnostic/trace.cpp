#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "RocketSim.h"
#include "bullet3-3.24/BulletCollision/CollisionShapes/btConvexPolyhedron.h"
#include "bullet3-3.24/BulletCollision/CollisionShapes/btTriangleShape.h"
#include "bullet3-3.24/BulletCollision/CollisionDispatch/btManifoldResult.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btGjkEpa2.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h"
#include "bullet3-3.24/BulletDynamics/Vehicle/btDefaultVehicleRaycaster.h"

using namespace RocketSim;

extern btScalar gContactBreakingThreshold;

namespace {

void PrintVec(const btVector3& value);

Car* traceCar = nullptr;
Arena* traceArena = nullptr;
ContactAddedCallback priorContactAddedCallback = nullptr;
std::vector<std::pair<int, int>> customGjkProbes;
int traceTick = 0;
btSingleConstraintRowSolver priorGenericRowSolver = nullptr;
btSingleConstraintRowSolver priorLowerLimitRowSolver = nullptr;
btSingleConstraintRowSolver priorSplitRowSolver = nullptr;
int solverRowCall = 0;

btScalar TraceSolverRow(
    const char* kind,
    btSingleConstraintRowSolver original,
    btSolverBody& bodyA,
    btSolverBody& bodyB,
    const btSolverConstraint& constraint) {
    const bool carIsA = traceCar && bodyA.m_originalBody == &traceCar->_rigidBody;
    const bool carIsB = traceCar && bodyB.m_originalBody == &traceCar->_rigidBody;
    const int call = solverRowCall++;
    if (!carIsA && !carIsB) {
        return original(bodyA, bodyB, constraint);
    }
    btSolverBody& carBody = carIsA ? bodyA : bodyB;
    const btVector3 deltaLinearBefore = carBody.m_deltaLinearVelocity;
    const btVector3 deltaAngularBefore = carBody.m_deltaAngularVelocity;
    const btVector3 pushBefore = carBody.m_pushVelocity;
    const btVector3 turnBefore = carBody.m_turnVelocity;
    const btScalar appliedBefore = constraint.m_appliedImpulse;
    const btScalar appliedPushBefore = constraint.m_appliedPushImpulse;
    const btScalar result = original(bodyA, bodyB, constraint);
    int face = -1;
    if (constraint.m_originalContactPoint) {
        face = static_cast<const btManifoldPoint*>(
            constraint.m_originalContactPoint)->m_index1;
    }
    std::cout << "{\"record\":\"solver_row\",\"tick\":" << traceTick
              << ",\"call\":" << call << ",\"kind\":\"" << kind
              << "\",\"face\":" << face
              << ",\"car_side\":\"" << (carIsA ? "a" : "b")
              << "\",\"friction_index\":" << constraint.m_frictionIndex
              << ",\"normal_1\":";
    PrintVec(constraint.m_contactNormal1);
    std::cout << ",\"normal_2\":";
    PrintVec(constraint.m_contactNormal2);
    std::cout << ",\"rel_cross_1\":";
    PrintVec(constraint.m_relpos1CrossNormal);
    std::cout << ",\"rel_cross_2\":";
    PrintVec(constraint.m_relpos2CrossNormal);
    std::cout << ",\"angular_component_a\":";
    PrintVec(constraint.m_angularComponentA);
    std::cout << ",\"angular_component_b\":";
    PrintVec(constraint.m_angularComponentB);
    std::cout << ",\"jacobian_inverse\":" << constraint.m_jacDiagABInv
              << ",\"rhs\":" << constraint.m_rhs
              << ",\"rhs_penetration\":" << constraint.m_rhsPenetration
              << ",\"cfm\":" << constraint.m_cfm
              << ",\"lower\":" << constraint.m_lowerLimit
              << ",\"upper\":" << constraint.m_upperLimit
              << ",\"friction\":" << constraint.m_friction
              << ",\"applied_before\":" << appliedBefore
              << ",\"applied_after\":" << btScalar(constraint.m_appliedImpulse)
              << ",\"push_before\":" << appliedPushBefore
              << ",\"push_after\":"
              << btScalar(constraint.m_appliedPushImpulse)
              << ",\"result\":" << result
              << ",\"base_linear\":";
    PrintVec(carBody.m_linearVelocity);
    std::cout << ",\"base_angular\":";
    PrintVec(carBody.m_angularVelocity);
    std::cout << ",\"external_force_impulse\":";
    PrintVec(carBody.m_externalForceImpulse);
    std::cout << ",\"external_torque_impulse\":";
    PrintVec(carBody.m_externalTorqueImpulse);
    std::cout << ",\"delta_linear_before\":";
    PrintVec(deltaLinearBefore);
    std::cout << ",\"delta_linear_after\":";
    PrintVec(carBody.m_deltaLinearVelocity);
    std::cout << ",\"delta_angular_before\":";
    PrintVec(deltaAngularBefore);
    std::cout << ",\"delta_angular_after\":";
    PrintVec(carBody.m_deltaAngularVelocity);
    std::cout << ",\"push_velocity_before\":";
    PrintVec(pushBefore);
    std::cout << ",\"push_velocity_after\":";
    PrintVec(carBody.m_pushVelocity);
    std::cout << ",\"turn_velocity_before\":";
    PrintVec(turnBefore);
    std::cout << ",\"turn_velocity_after\":";
    PrintVec(carBody.m_turnVelocity);
    std::cout << "}\n";
    return result;
}

btScalar TraceGenericSolverRow(
    btSolverBody& bodyA,
    btSolverBody& bodyB,
    const btSolverConstraint& constraint) {
    return TraceSolverRow(
        "generic", priorGenericRowSolver, bodyA, bodyB, constraint);
}

btScalar TraceLowerLimitSolverRow(
    btSolverBody& bodyA,
    btSolverBody& bodyB,
    const btSolverConstraint& constraint) {
    return TraceSolverRow(
        "lower_limit", priorLowerLimitRowSolver, bodyA, bodyB, constraint);
}

btScalar TraceSplitSolverRow(
    btSolverBody& bodyA,
    btSolverBody& bodyB,
    const btSolverConstraint& constraint) {
    return TraceSolverRow("split", priorSplitRowSolver, bodyA, bodyB, constraint);
}

class TracingVehicleRaycaster final : public btDefaultVehicleRaycaster {
public:
    explicit TracingVehicleRaycaster(btDynamicsWorld* world)
        : btDefaultVehicleRaycaster(world) {}

    void* castRay(
        const btVector3& from,
        const btVector3& to,
        const btCollisionObject* ignoreObject,
        btVehicleRaycasterResult& result) override {
        if (traceTick != lastTick) {
            lastTick = traceTick;
            nextWheel = 0;
        }
        const int wheel = nextWheel++;
        void* object = btDefaultVehicleRaycaster::castRay(
            from, to, ignoreObject, result);
        btScalar realRayLength = btScalar(0);
        btVector3 wheelDirection(0, 0, 0);
        if (traceCar && wheel < traceCar->_bulletVehicle.getNumWheels()) {
            const btWheelInfoRL& wheelInfo =
                traceCar->_bulletVehicle.getWheelInfo(wheel);
            const btScalar suspensionTravel =
                wheelInfo.m_maxSuspensionTravelCm / btScalar(100);
            realRayLength = wheelInfo.getSuspensionRestLength()
                + suspensionTravel + wheelInfo.m_wheelsRadius
                - RLConst::BTVehicle::SUSPENSION_SUBTRACTION;
            wheelDirection = wheelInfo.m_raycastInfo.m_wheelDirectionWS;
        }
        std::cout << "{\"record\":\"vehicle_ray\",\"tick\":" << traceTick
                  << ",\"wheel\":" << wheel << ",\"from_bt\":";
        PrintVec(from);
        std::cout << ",\"to_bt\":";
        PrintVec(to);
        std::cout << ",\"direction\":";
        PrintVec(wheelDirection);
        std::cout << ",\"real_ray_length_bt\":" << realRayLength;
        std::cout << ",\"hit\":" << (object ? "true" : "false");
        if (object) {
            std::cout << ",\"fraction\":" << result.m_distFraction
                      << ",\"point_bt\":";
            PrintVec(result.m_hitPointInWorld);
            std::cout << ",\"normal\":";
            PrintVec(result.m_hitNormalInWorld);
        }
        std::cout << "}\n";
        return object;
    }

private:
    int lastTick = -1;
    int nextWheel = 0;
};

struct BvhInspector final : btOptimizedBvh {
    static const btVector3& Minimum(const btOptimizedBvh* value) {
        return reinterpret_cast<const BvhInspector*>(value)->m_bvhAabbMin;
    }
    static const btVector3& Maximum(const btOptimizedBvh* value) {
        return reinterpret_cast<const BvhInspector*>(value)->m_bvhAabbMax;
    }
    static const btVector3& Quantization(const btOptimizedBvh* value) {
        return reinterpret_cast<const BvhInspector*>(value)->m_bvhQuantization;
    }
};

bool TraceContactAdded(
    btManifoldPoint& point,
    const btCollisionObjectWrapper* objectA,
    int partA,
    int indexA,
    const btCollisionObjectWrapper* objectB,
    int partB,
    int indexB) {
    const btVector3 rawNormalB = point.m_normalWorldOnB;
    const btVector3 rawLocalA = point.m_localPointA;
    const btVector3 rawLocalB = point.m_localPointB;
    const btVector3 rawPointA = point.m_positionWorldOnA;
    const btVector3 rawPointB = point.m_positionWorldOnB;
    const btScalar rawDistance = point.getDistance();
    const bool result = priorContactAddedCallback
        ? priorContactAddedCallback(
              point, objectA, partA, indexA, objectB, partB, indexB)
        : true;
    const bool involvesTraceCar = traceCar
        && (objectA->getCollisionObject() == &traceCar->_rigidBody
            || objectB->getCollisionObject() == &traceCar->_rigidBody);
    if (!involvesTraceCar || !traceArena) {
        return result;
    }
    const btCollisionObject* worldBody =
        objectA->getCollisionObject() == &traceCar->_rigidBody
        ? objectB->getCollisionObject()
        : objectA->getCollisionObject();
    int worldBodyIndex = -1;
    for (int index = 0; index < traceArena->_worldCollisionRBs.size(); ++index) {
        if (traceArena->_worldCollisionRBs[index] == worldBody) {
            worldBodyIndex = index;
            break;
        }
    }
    std::cout << "{\"record\":\"contact_added\",\"tick\":" << traceTick
              << ",\"world_body_index\":" << worldBodyIndex
              << ",\"face\":" << point.m_index1
              << ",\"raw_distance_bt\":" << rawDistance
              << ",\"raw_local_a\":";
    PrintVec(rawLocalA);
    std::cout << ",\"raw_local_b\":";
    PrintVec(rawLocalB);
    std::cout << ",\"raw_point_a\":";
    PrintVec(rawPointA);
    std::cout << ",\"raw_point_b\":";
    PrintVec(rawPointB);
    std::cout << ",\"raw_normal_b\":";
    PrintVec(rawNormalB);
    std::cout << ",\"adjusted_distance_bt\":" << point.getDistance()
              << ",\"adjusted_local_a\":";
    PrintVec(point.m_localPointA);
    std::cout << ",\"adjusted_local_b\":";
    PrintVec(point.m_localPointB);
    std::cout << ",\"adjusted_normal_b\":";
    PrintVec(point.m_normalWorldOnB);
    const auto* meshShape = dynamic_cast<const btBvhTriangleMeshShape*>(
        worldBody->getCollisionShape());
    const btTriangleInfo* triangleInfo = nullptr;
    if (meshShape && meshShape->getTriangleInfoMap() && point.m_index1 >= 0) {
        triangleInfo = meshShape->getTriangleInfoMap()->find(
            btHashInt(point.m_index1));
    }
    std::cout << ",\"edge_info\":";
    if (triangleInfo) {
        std::cout << "{\"angles\":[" << triangleInfo->m_edgeV0V1Angle
                  << ',' << triangleInfo->m_edgeV1V2Angle << ','
                  << triangleInfo->m_edgeV2V0Angle << "],\"flags\":"
                  << triangleInfo->m_flags << '}';
    } else {
        std::cout << "null";
    }
    std::cout << "}\n";
    auto* dispatcher = traceArena->_bulletWorld.getDispatcher();
    for (int manifoldIndex = 0; manifoldIndex < dispatcher->getNumManifolds();
         ++manifoldIndex) {
        btPersistentManifold* manifold =
            dispatcher->getManifoldByIndexInternal(manifoldIndex);
        const bool matches =
            (manifold->getBody0() == &traceCar->_rigidBody
             && manifold->getBody1() == worldBody)
            || (manifold->getBody1() == &traceCar->_rigidBody
                && manifold->getBody0() == worldBody);
        if (!matches) {
            continue;
        }
        std::cout << "{\"record\":\"manifold_after_add\",\"tick\":"
                  << traceTick << ",\"world_body_index\":" << worldBodyIndex
                  << ",\"candidate_face\":" << point.m_index1
                  << ",\"faces\":[";
        for (int cacheIndex = 0; cacheIndex < manifold->getNumContacts(); ++cacheIndex) {
            if (cacheIndex) {
                std::cout << ',';
            }
            std::cout << manifold->getContactPoint(cacheIndex).m_index1;
        }
        std::cout << "]}\n";
        break;
    }
    return result;
}

struct Scenario {
    Vec pos;
    Vec vel;
    Vec angVel;
    RotMat rotMat;
    CarControls controls;
    bool onGround;
    float boost;
    bool seedLastControls;
};

Scenario MakeScenario(const std::string& name) {
    Scenario result = {};
    result.rotMat = RotMat(
        Vec(1.0f, 0.0f, 0.0f),
        Vec(0.0f, 1.0f, 0.0f),
        Vec(0.0f, 0.0f, 1.0f));
    result.boost = 100.0f;
    if (name == "floor_rest") {
        result.pos = Vec(0, 0, 17);
        result.onGround = true;
    } else if (name == "throttle_forward") {
        result.pos = Vec(0, 0, 17);
        result.controls.throttle = 1.0f;
        result.onGround = true;
    } else if (name == "throttle_reverse") {
        result.pos = Vec(0, 0, 17);
        result.controls.throttle = -1.0f;
        result.onGround = true;
    } else if (name == "brake_to_reverse") {
        result.pos = Vec(0, 0, 17);
        result.vel = Vec(1000, 0, 0);
        result.controls.throttle = -1.0f;
        result.onGround = true;
    } else if (name == "ground_boost") {
        result.pos = Vec(0, 0, 17);
        result.controls.throttle = 1.0f;
        result.controls.boost = true;
        result.onGround = true;
        result.boost = 80.0f;
    } else if (name == "steer_low_full_left") {
        result.pos = Vec(0, 0, 17);
        result.vel = Vec(250, 0, 0);
        result.controls.throttle = 0.5f;
        result.controls.steer = 1.0f;
        result.onGround = true;
    } else if (name == "powerslide_initiation") {
        result.pos = Vec(0, 0, 17);
        result.vel = Vec(900, 0, 0);
        result.controls.throttle = 0.5f;
        result.controls.steer = 1.0f;
        result.controls.handbrake = true;
        result.onGround = true;
    } else if (name == "ramp_transition") {
        result.pos = Vec(3800, 4300, 120);
        result.vel = Vec(500, 500, -100);
        // Match the float32 quaternion-to-matrix conversion used by the
        // Python parity oracle.  Recomputing from Euler angles is close, but
        // a few ULPs are enough to choose a different late ramp trajectory.
        result.rotMat = RotMat(
            Vec(0.78847325f, 0.53942358f, 0.29552022f),
            Vec(-0.56464255f, 0.82533562f, -7.4505806e-09f),
            Vec(-0.24390337f, -0.16686326f, 0.95533651f));
        result.controls.throttle = 0.8f;
        result.controls.steer = 0.4f;
    } else if (name == "ramp_sim_tick533") {
        // Diagnostic replay of RivalSim's post-tick-533 transform.  This is
        // used only to ask the pinned Bullet narrowphase how it classifies
        // the exact CUDA-side state at the late ramp breaking boundary.
        result.pos = Vec(1576.0220947265625f, 5162.2109375f, 15.90343952178955f);
        result.vel = Vec(-184.17227172851562f, -64.72840118408203f, 1.2372932434082031f);
        result.rotMat = RotMat(
            Vec(-0.21895063f, -0.97555166f, -0.01897352f),
            Vec(0.9753416f, -0.21826768f, -0.03268551f),
            Vec(0.02774509f, -0.02566217f, 0.9992856f));
    } else if (name == "side_wall_transition") {
        result.pos = Vec(3920, 0, 260);
        result.vel = Vec(700, 100, -50);
        result.rotMat = RotMat(
            Vec(0.87758255f, 0.0f, 0.47942555f),
            Vec(0.0f, 1.0f, 0.0f),
            Vec(-0.47942555f, 0.0f, 0.87758255f));
        result.controls.throttle = 1.0f;
    } else if (name == "back_wall_transition") {
        result.pos = Vec(700, 5050, 300);
        result.vel = Vec(100, 700, -40);
        result.controls.throttle = 1.0f;
    } else if (name == "ceiling_contact") {
        result.pos = Vec(0, 0, 2028);
        result.vel = Vec(200, 0, 80);
        result.rotMat = RotMat(
            Vec(1.0f, 0.0f, 0.0f),
            Vec(0.0f, -1.0f, 1.2246469e-16f),
            Vec(0.0f, -1.2246469e-16f, -1.0f));
        result.controls.throttle = 0.5f;
    } else if (name == "corner_transition") {
        result.pos = Vec(3800, 4700, 180);
        result.vel = Vec(500, 500, -100);
        result.rotMat = RotMat(
            Vec(0.74106509f, 0.62419051f, 0.24740395f),
            Vec(-0.66896945f, 0.71793193f, 0.19249319f),
            Vec(-0.057466768f, -0.30815566f, 0.94959867f));
        result.controls.throttle = 0.8f;
        result.controls.steer = -0.5f;
    } else if (name == "roof_impact") {
        result.pos = Vec(0, 0, 22);
        result.vel = Vec(100, 0, -500);
        // Match the float32 matrix passed through the Python parity oracle.
        // Angle(float(pi)) instead produces sin(pi) ~= -8.74e-8 and sends
        // this deliberately chaotic corner-impact case to another trajectory.
        result.rotMat = RotMat(
            Vec(1.0f, 0.0f, 0.0f),
            Vec(0.0f, -1.0f, 1.2246469e-16f),
            Vec(0.0f, -1.2246469e-16f, -1.0f));
        result.onGround = true;
    } else if (name == "off_center_impact") {
        result.pos = Vec(0, 0, 5);
        result.vel = Vec(500, 300, -350);
        result.rotMat = RotMat(
            Vec(0.86008930f, 0.17434874f, -0.47942555f),
            Vec(-1.0982156e-05f, 0.93979210f, 0.34174675f),
            Vec(0.51014346f, -0.29392749f, 0.80830705f));
        result.onGround = true;
    } else if (name == "side_impact") {
        result.pos = Vec(0, 0, 10);
        result.vel = Vec(200, 800, -200);
        result.rotMat = RotMat(
            Vec(1.0f, 0.0f, 0.0f),
            Vec(0.0f, 0.69670671f, 0.71735609f),
            Vec(0.0f, -0.71735609f, 0.69670671f));
        result.onGround = true;
    } else if (name == "wall_scrape") {
        result.pos = Vec(4075, 1500, 650);
        result.vel = Vec(100, 900, -50);
        result.rotMat = RotMat(
            Vec(5.9604645e-08f, 0.0f, 0.99999994f),
            Vec(-0.38941833f, 0.92106098f, 0.0f),
            Vec(-0.92106098f, -0.38941833f, 5.9604645e-08f));
        result.controls.throttle = 0.4f;
        result.controls.steer = 0.5f;
    } else {
        throw std::runtime_error("unknown scenario: " + name);
    }
    return result;
}

Scenario MakeCustomScenario(int argc, char** argv) {
    constexpr int CUSTOM_VALUE_COUNT = 21;
    constexpr int CUSTOM_BASE_ARGC = 4 + CUSTOM_VALUE_COUNT;
    if (argc < CUSTOM_BASE_ARGC || (argc - CUSTOM_BASE_ARGC) % 2 != 0) {
        throw std::runtime_error(
            "custom scenario requires 21 values: pos3 vel3 ang_vel3 "
            "forward3 right3 up3 throttle steer handbrake, followed by "
            "optional GJK probe body/face pairs");
    }
    int index = 4;
    const auto vector = [&]() {
        const Vec result(
            std::stof(argv[index]),
            std::stof(argv[index + 1]),
            std::stof(argv[index + 2]));
        index += 3;
        return result;
    };
    Scenario result = {};
    result.pos = vector();
    result.vel = vector();
    result.angVel = vector();
    const Vec forward = vector();
    const Vec right = vector();
    const Vec up = vector();
    result.rotMat = RotMat(forward, right, up);
    result.controls.throttle = std::stof(argv[index++]);
    result.controls.steer = std::stof(argv[index++]);
    result.controls.handbrake = std::stoi(argv[index++]) != 0;
    result.boost = 100.0f;
    result.seedLastControls = true;
    customGjkProbes.clear();
    while (index < argc) {
        customGjkProbes.emplace_back(
            std::stoi(argv[index]), std::stoi(argv[index + 1]));
        index += 2;
    }
    return result;
}

void PrintVec(const btVector3& value) {
    std::cout << '[' << value.x() << ',' << value.y() << ',' << value.z() << ']';
}

void PrintVec(const Vec& value) {
    std::cout << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}

void PrintQuat(const btQuaternion& value) {
    std::cout << '[' << value.x() << ',' << value.y() << ',' << value.z() << ','
              << value.w() << ']';
}

struct FaceTraversalCallback final : btTriangleCallback {
    std::vector<int> faces;

    void processTriangle(btVector3*, int, int triangleIndex) override {
        faces.push_back(triangleIndex);
    }
};

void PrintBvhTraversal(Car* car, Arena* arena, int tick) {
    const btTransform childWorld = car->_rigidBody.getWorldTransform()
        * car->_compoundShape.getChildTransform(0);
    btVector3 minimum;
    btVector3 maximum;
    car->_childHitboxShape.getAabb(childWorld, minimum, maximum);
    for (int bodyIndex = 0; bodyIndex < arena->_worldCollisionRBs.size(); ++bodyIndex) {
        auto* bvh = dynamic_cast<btBvhTriangleMeshShape*>(
            arena->_worldCollisionRBs[bodyIndex]->getCollisionShape());
        if (!bvh) {
            continue;
        }
        if (tick == 1) {
            btOptimizedBvh* optimized = bvh->getOptimizedBvh();
            std::cout << "{\"record\":\"bvh_layout\",\"tick\":" << tick
                      << ",\"world_body_index\":" << bodyIndex
                      << ",\"local_aabb_min\":";
            PrintVec(bvh->getLocalAabbMin());
            std::cout << ",\"local_aabb_max\":";
            PrintVec(bvh->getLocalAabbMax());
            std::cout << ",\"bvh_min\":";
            PrintVec(BvhInspector::Minimum(optimized));
            std::cout << ",\"bvh_max\":";
            PrintVec(BvhInspector::Maximum(optimized));
            std::cout << ",\"quantization\":";
            PrintVec(BvhInspector::Quantization(optimized));
            std::cout << ",\"leaf_order\":[";
            const auto& nodes = optimized->getQuantizedNodeArray();
            bool firstLeaf = true;
            for (int nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
                if (!nodes[nodeIndex].isLeafNode()) {
                    continue;
                }
                if (!firstLeaf) {
                    std::cout << ',';
                }
                firstLeaf = false;
                std::cout << nodes[nodeIndex].getTriangleIndex();
            }
            FaceTraversalCallback fullCallback;
            bvh->processAllTriangles(
                &fullCallback, bvh->getLocalAabbMin(), bvh->getLocalAabbMax());
            std::cout << "],\"cache_order\":[";
            for (int index = 0; index < fullCallback.faces.size(); ++index) {
                if (index) {
                    std::cout << ',';
                }
                std::cout << fullCallback.faces[index];
            }
            std::cout << "]}\n";
        }
        FaceTraversalCallback callback;
        bvh->processAllTriangles(&callback, minimum, maximum);
        if (callback.faces.empty()) {
            continue;
        }
        std::cout << "{\"record\":\"bvh_traversal\",\"tick\":" << tick
                  << ",\"world_body_index\":" << bodyIndex << ",\"faces\":[";
        for (int index = 0; index < callback.faces.size(); ++index) {
            if (index) {
                std::cout << ',';
            }
            std::cout << callback.faces[index];
        }
        std::cout << "]}\n";
    }
}

void PrintMatrix(const btMatrix3x3& value) {
    std::cout << '[';
    for (int row = 0; row < 3; ++row) {
        if (row) {
            std::cout << ',';
        }
        PrintVec(value[row]);
    }
    std::cout << ']';
}

void PrintWheelApplyReplay(
    Car* car,
    const btVector3& initialLinear,
    const btVector3& initialAngular,
    btScalar timeStep,
    int tick) {
    btRigidBody& body = car->_rigidBody;
    btVector3 linear = initialLinear;
    btVector3 angular = initialAngular;
    const btVector3 origin = body.getWorldTransform().getOrigin();
    const btMatrix3x3 inverseInertia = body.getInvInertiaTensorWorld();
    const btVector3 linearFactor = body.getLinearFactor();
    const btVector3 angularFactor = body.getAngularFactor();
    const btScalar inverseMass = body.getInvMass();
    const auto apply = [&](const btVector3& impulse, const btVector3& relative) {
        linear += impulse * linearFactor * inverseMass;
        angular += inverseInertia * relative.cross(impulse * linearFactor)
            * angularFactor;
    };

    std::cout << "{\"record\":\"wheel_apply_replay\",\"tick\":" << tick
              << ",\"initial_linear_bt\":";
    PrintVec(initialLinear);
    std::cout << ",\"initial_angular\":";
    PrintVec(initialAngular);
    std::cout << ",\"stages\":[";
    bool first = true;
    for (int wheelIndex = 0; wheelIndex < car->_bulletVehicle.getNumWheels();
         ++wheelIndex) {
        const btWheelInfoRL& wheel = car->_bulletVehicle.getWheelInfo(wheelIndex);
        const bool active = wheel.m_wheelsSuspensionForce != 0;
        const btScalar scale = active
            ? wheel.m_wheelsSuspensionForce * timeStep + wheel.m_extraPushback
            : btScalar(0);
        const btVector3 impulse = active
            ? wheel.m_raycastInfo.m_contactNormalWS * scale
            : btVector3(0, 0, 0);
        const btVector3 relative = wheel.m_raycastInfo.m_contactPointWS - origin;
        if (active) {
            apply(impulse, relative);
        }
        if (!first) {
            std::cout << ',';
        }
        first = false;
        std::cout << "{\"kind\":\"suspension\",\"wheel\":" << wheelIndex
                  << ",\"active\":" << (active ? "true" : "false")
                  << ",\"scale\":" << scale << ",\"impulse_bt\":";
        PrintVec(impulse);
        std::cout << ",\"relative_bt\":";
        PrintVec(relative);
        std::cout << ",\"linear_after_bt\":";
        PrintVec(linear);
        std::cout << ",\"angular_after\":";
        PrintVec(angular);
        std::cout << '}';
    }

    const btVector3 up = body.getWorldTransform().getBasis().getColumn(2);
    for (int wheelIndex = 0; wheelIndex < car->_bulletVehicle.getNumWheels();
         ++wheelIndex) {
        const btWheelInfoRL& wheel = car->_bulletVehicle.getWheelInfo(wheelIndex);
        const bool active = !wheel.m_impulse.isZero();
        const btVector3 rawAxle =
            wheel.m_worldTransform.getBasis().getColumn(1);
        const btVector3 surfaceNormal = wheel.m_raycastInfo.m_contactNormalWS;
        const btScalar axleProjection = rawAxle.dot(surfaceNormal);
        const btVector3 projectedAxle =
            rawAxle - surfaceNormal * axleProjection;
        const btVector3 axle = projectedAxle.safeNormalized();
        const btVector3 forward = surfaceNormal.cross(axle).safeNormalized();
        btVector3 relative = wheel.m_raycastInfo.m_contactPointWS - origin;
        const btVector3 pointVelocity =
            initialLinear + initialAngular.cross(relative);
        const btScalar lateralVelocity = axle.dot(pointVelocity);
        const btScalar forwardVelocity = forward.dot(pointVelocity);
        btScalar jacobianDiagonal = 0;
        btScalar jacobianInverse = 0;
        btScalar sideImpulse = 0;
        const auto* ground = static_cast<const btRigidBody*>(
            wheel.m_raycastInfo.m_groundObject);
        if (ground) {
            const btVector3 groundRelative =
                wheel.m_raycastInfo.m_contactPointWS
                - ground->getCenterOfMassPosition();
            btJacobianEntry jacobian(
                body.getCenterOfMassTransform().getBasis().transpose(),
                ground->getCenterOfMassTransform().getBasis().transpose(),
                relative,
                groundRelative,
                axle,
                body.getInvInertiaDiagLocal(),
                body.getInvMass(),
                ground->getInvInertiaDiagLocal(),
                ground->getInvMass());
            jacobianDiagonal = jacobian.getDiagonal();
            jacobianInverse = btScalar(1) / jacobianDiagonal;
            sideImpulse = -btScalar(0.2) * lateralVelocity * jacobianInverse;
        }
        const btScalar rollingUnclamped =
            -forwardVelocity * btScalar(113.73963f);
        relative -= up * up.dot(relative);
        const btVector3 impulse = active
            ? wheel.m_impulse * timeStep
            : btVector3(0, 0, 0);
        if (active) {
            apply(impulse, relative);
        }
        std::cout << ",{\"kind\":\"friction\",\"wheel\":" << wheelIndex
                  << ",\"active\":" << (active ? "true" : "false")
                  << ",\"raw_axle\":";
        PrintVec(rawAxle);
        std::cout << ",\"surface_normal\":";
        PrintVec(surfaceNormal);
        std::cout << ",\"axle_projection\":" << axleProjection
                  << ",\"projected_axle\":";
        PrintVec(projectedAxle);
        std::cout << ",\"axle\":";
        PrintVec(axle);
        std::cout << ",\"forward\":";
        PrintVec(forward);
        std::cout << ",\"point_velocity_bt\":";
        PrintVec(pointVelocity);
        std::cout << ",\"lateral_velocity\":" << lateralVelocity
                  << ",\"forward_velocity\":" << forwardVelocity
                  << ",\"jacobian_diagonal\":" << jacobianDiagonal
                  << ",\"jacobian_inverse\":" << jacobianInverse
                  << ",\"side_impulse_source\":" << sideImpulse
                  << ",\"rolling_unclamped\":" << rollingUnclamped
                  << ",\"impulse_bt\":";
        PrintVec(impulse);
        std::cout << ",\"relative_bt\":";
        PrintVec(relative);
        std::cout << ",\"linear_after_bt\":";
        PrintVec(linear);
        std::cout << ",\"angular_after\":";
        PrintVec(angular);
        std::cout << '}';
    }
    std::cout << "],\"replayed_linear_bt\":";
    PrintVec(linear);
    std::cout << ",\"actual_linear_bt\":";
    PrintVec(body.getLinearVelocity());
    std::cout << ",\"replayed_angular\":";
    PrintVec(angular);
    std::cout << ",\"actual_angular\":";
    PrintVec(body.getAngularVelocity());
    std::cout << "}\n";
}

void PrintState(const char* phase, int tick, Car* car, Arena* arena) {
    const CarState state = car->GetState();
    std::cout << "{\"record\":\"state\",\"phase\":\"" << phase
              << "\",\"tick\":" << tick << ",\"pos\":";
    PrintVec(state.pos);
    std::cout << ",\"vel\":";
    PrintVec(state.vel);
    std::cout << ",\"boost\":" << state.boost;
    std::cout << ",\"ang_vel\":";
    PrintVec(state.angVel);
    std::cout << ",\"gyro_impulse\":";
    PrintVec(car->_rigidBody.computeGyroscopicImpulseImplicit_Body(1.0f / 120.0f));
    std::cout << ",\"rigid_pos_bt\":";
    PrintVec(car->_rigidBody.getWorldTransform().getOrigin());
    std::cout << ",\"rigid_quat\":";
    PrintQuat(car->_rigidBody.getWorldTransform().getRotation());
    std::cout << ",\"rigid_vel_bt\":";
    PrintVec(car->_rigidBody.getLinearVelocity());
    std::cout << ",\"rigid_ang_vel\":";
    PrintVec(car->_rigidBody.getAngularVelocity());
    std::cout << ",\"rigid_flags\":" << car->_rigidBody.getFlags();
    std::cout << ",\"total_force_bt\":";
    PrintVec(car->_rigidBody.getTotalForce());
    std::cout << ",\"total_torque_bt\":";
    PrintVec(car->_rigidBody.getTotalTorque());
    std::cout << ",\"inv_inertia_world\":";
    PrintMatrix(car->_rigidBody.getInvInertiaTensorWorld());
    std::cout << ",\"on_ground\":" << (state.isOnGround ? "true" : "false")
              << ",\"world_contact\":" << (state.worldContact.hasContact ? "true" : "false")
              << ",\"world_normal\":";
    PrintVec(state.worldContact.contactNormal);
    std::cout << ",\"wheels\":[";
    for (int index = 0; index < car->_bulletVehicle.getNumWheels(); ++index) {
        if (index) {
            std::cout << ',';
        }
        const btWheelInfoRL& wheel = car->_bulletVehicle.getWheelInfo(index);
        std::cout << "{\"contact\":"
                  << (wheel.m_raycastInfo.m_isInContact ? "true" : "false")
                  << ",\"world\":" << (wheel.m_isInContactWithWorld ? "true" : "false")
                  << ",\"length\":" << wheel.m_raycastInfo.m_suspensionLength
                  << ",\"relative_velocity\":" << wheel.m_suspensionRelativeVelocity
                  << ",\"clipped_inv\":" << wheel.m_clippedInvContactDotSuspension
                  << ",\"suspension_force\":" << wheel.m_wheelsSuspensionForce
                  << ",\"extra_pushback\":" << wheel.m_extraPushback
                  << ",\"engine_force\":" << wheel.m_engineForce
                  << ",\"brake\":" << wheel.m_brake
                  << ",\"steering\":" << wheel.m_steering
                  << ",\"lat_friction\":" << wheel.m_latFriction
                  << ",\"long_friction\":" << wheel.m_longFriction
                  << ",\"side_impulse\":"
                  << (index < car->_bulletVehicle.m_sideImpulse.size()
                          ? car->_bulletVehicle.m_sideImpulse[index]
                          : 0.0f)
                  << ",\"forward_impulse\":"
                  << (index < car->_bulletVehicle.m_forwardImpulse.size()
                          ? car->_bulletVehicle.m_forwardImpulse[index]
                          : 0.0f)
                  << ",\"impulse\":";
        PrintVec(wheel.m_impulse);
        std::cout << ",\"hard_point\":";
        PrintVec(wheel.m_raycastInfo.m_hardPointWS);
        std::cout << ",\"point\":";
        PrintVec(wheel.m_raycastInfo.m_contactPointWS);
        std::cout << ",\"normal\":";
        PrintVec(wheel.m_raycastInfo.m_contactNormalWS);
        std::cout << ",\"basis_column_0\":";
        PrintVec(wheel.m_worldTransform.getBasis().getColumn(0));
        std::cout << ",\"basis_column_1\":";
        PrintVec(wheel.m_worldTransform.getBasis().getColumn(1));
        std::cout << '}';
    }
    std::cout << "],\"manifolds\":[";

    auto* dispatcher = arena->_bulletWorld.getDispatcher();
    bool firstPoint = true;
    for (int manifoldIndex = 0; manifoldIndex < dispatcher->getNumManifolds(); ++manifoldIndex) {
        btPersistentManifold* manifold = dispatcher->getManifoldByIndexInternal(manifoldIndex);
        const bool involvesCar = manifold->getBody0() == &car->_rigidBody
            || manifold->getBody1() == &car->_rigidBody;
        if (!involvesCar) {
            continue;
        }
        const btCollisionObject* worldBody = manifold->getBody0() == &car->_rigidBody
            ? manifold->getBody1()
            : manifold->getBody0();
        int worldBodyIndex = -1;
        for (int index = 0; index < arena->_worldCollisionRBs.size(); ++index) {
            if (arena->_worldCollisionRBs[index] == worldBody) {
                worldBodyIndex = index;
                break;
            }
        }
        for (int pointIndex = 0; pointIndex < manifold->getNumContacts(); ++pointIndex) {
            if (!firstPoint) {
                std::cout << ',';
            }
            firstPoint = false;
            const btManifoldPoint& point = manifold->getContactPoint(pointIndex);
            const btTransform childWorld = car->_rigidBody.getWorldTransform()
                * car->_compoundShape.getChildTransform(0);
            int worldFaces = -1;
            int worldVertices = -1;
            btVector3 triangle[3] = {};
            const auto* bvh = dynamic_cast<const btBvhTriangleMeshShape*>(
                worldBody->getCollisionShape());
            if (bvh && point.m_index1 >= 0) {
                const unsigned char* vertexBase = nullptr;
                const unsigned char* indexBase = nullptr;
                int vertexStride = 0;
                int indexStride = 0;
                bvh->getMeshInterface()->getLockedReadOnlyVertexIndexBase(
                    &vertexBase, worldVertices, vertexStride, &indexBase, indexStride,
                    worldFaces);
                const auto* indices = reinterpret_cast<const int*>(
                    indexBase + point.m_index1 * indexStride);
                for (int corner = 0; corner < 3; ++corner) {
                    const auto* values = reinterpret_cast<const btScalar*>(
                        vertexBase + indices[corner] * vertexStride);
                    triangle[corner] = btVector3(values[0], values[1], values[2]);
                }
                bvh->getMeshInterface()->unLockReadOnlyVertexBase(0);
            }
            std::cout << "{\"distance_bt\":" << point.getDistance()
                      << ",\"breaking_threshold_bt\":"
                      << manifold->getContactBreakingThreshold()
                      << ",\"world_body_index\":" << worldBodyIndex
                      << ",\"world_faces\":" << worldFaces
                      << ",\"world_vertices\":" << worldVertices
                      << ",\"lifetime\":" << point.getLifeTime()
                      << ",\"applied_impulse\":" << point.m_appliedImpulse
                      << ",\"lateral_impulse_1\":" << point.m_appliedImpulseLateral1
                      << ",\"lateral_impulse_2\":" << point.m_appliedImpulseLateral2
                      << ",\"friction\":" << point.m_combinedFriction
                      << ",\"restitution\":" << point.m_combinedRestitution
                      << ",\"part_id_0\":" << point.m_partId0
                      << ",\"index_0\":" << point.m_index0
                      << ",\"part_id_1\":" << point.m_partId1
                      << ",\"index_1\":" << point.m_index1
                      << ",\"point_a\":";
            PrintVec(point.getPositionWorldOnA());
            std::cout << ",\"point_b\":";
            PrintVec(point.getPositionWorldOnB());
            std::cout << ",\"point_a_child_local\":";
            PrintVec(childWorld.inverse() * point.getPositionWorldOnA());
            std::cout << ",\"point_b_child_local\":";
            PrintVec(childWorld.inverse() * point.getPositionWorldOnB());
            std::cout << ",\"stored_local_a\":";
            PrintVec(point.m_localPointA);
            std::cout << ",\"stored_local_b\":";
            PrintVec(point.m_localPointB);
            std::cout << ",\"normal_b\":";
            PrintVec(point.m_normalWorldOnB);
            std::cout << ",\"friction_dir_1\":";
            PrintVec(point.m_lateralFrictionDir1);
            std::cout << ",\"friction_dir_2\":";
            PrintVec(point.m_lateralFrictionDir2);
            std::cout << ",\"triangle_bt\":[";
            PrintVec(triangle[0]);
            std::cout << ',';
            PrintVec(triangle[1]);
            std::cout << ',';
            PrintVec(triangle[2]);
            std::cout << ']';
            std::cout << '}';
        }
    }
    std::cout << "]}\n";
}

void PrintAirDampingReplay(int tick, Car* car) {
    using namespace RLConst;
    const btVector3 dirPitch = -car->GetRightDir();
    const btVector3 dirYaw = car->GetUpDir();
    const btVector3 dirRoll = -car->GetForwardDir();
    const btVector3 angularVelocity = car->_rigidBody.getAngularVelocity();
    const float dampPitch = dirPitch.dot(angularVelocity)
        * CAR_AIR_CONTROL_DAMPING.x;
    const float dampYaw = dirYaw.dot(angularVelocity)
        * CAR_AIR_CONTROL_DAMPING.y;
    const float dampRoll = dirRoll.dot(angularVelocity)
        * CAR_AIR_CONTROL_DAMPING.z;
    const btVector3 damping =
        (dirYaw * dampYaw)
        + (dirPitch * dampPitch)
        + (dirRoll * dampRoll);
    const btVector3 requested = -damping;
    const btMatrix3x3 inertiaWorld =
        car->_rigidBody.getInvInertiaTensorWorld().inverse();
    const btVector3 angularAcceleration = inertiaWorld * requested;
    const btVector3 applied = angularAcceleration * CAR_TORQUE_SCALE;
    std::cout << "{\"record\":\"air_damping_replay\",\"tick\":" << tick
              << ",\"dir_pitch\":";
    PrintVec(dirPitch);
    std::cout << ",\"dir_yaw\":";
    PrintVec(dirYaw);
    std::cout << ",\"dir_roll\":";
    PrintVec(dirRoll);
    std::cout << ",\"angular_velocity\":";
    PrintVec(angularVelocity);
    std::cout << ",\"damp_pitch\":" << dampPitch
              << ",\"damp_yaw\":" << dampYaw
              << ",\"damp_roll\":" << dampRoll
              << ",\"damping\":";
    PrintVec(damping);
    std::cout << ",\"requested\":";
    PrintVec(requested);
    std::cout << ",\"inertia_world\":";
    PrintMatrix(inertiaWorld);
    std::cout << ",\"angular_acceleration\":";
    PrintVec(angularAcceleration);
    std::cout << ",\"applied\":";
    PrintVec(applied);
    std::cout << "}\n";
}

void PrintShape(Car* car) {
    const btVector3 withMargin = car->_childHitboxShape.getHalfExtentsWithMargin();
    const btVector3 withoutMargin = car->_childHitboxShape.getHalfExtentsWithoutMargin();
    const btTransform child = car->_compoundShape.getChildTransform(0);
    std::cout << "{\"record\":\"shape\",\"half_with_margin\":";
    PrintVec(withMargin);
    std::cout << ",\"half_without_margin\":";
    PrintVec(withoutMargin);
    std::cout << ",\"margin\":" << car->_childHitboxShape.getMargin()
              << ",\"child_origin\":";
    PrintVec(child.getOrigin());
    std::cout << ",\"inv_inertia_local\":";
    PrintVec(car->_rigidBody.getInvInertiaDiagLocal());
    const btConvexPolyhedron* hull = car->_childHitboxShape.getConvexPolyhedron();
    std::cout << ",\"polyhedron_vertices\":[";
    if (hull) {
        for (int index = 0; index < hull->m_vertices.size(); ++index) {
            if (index) {
                std::cout << ',';
            }
            PrintVec(hull->m_vertices[index]);
        }
    }
    std::cout << ']';
    std::cout << ",\"inv_mass\":" << car->_rigidBody.getInvMass();
    std::cout << ",\"body_breaking_threshold_bt\":"
              << car->_rigidBody.getCollisionShape()->getContactBreakingThreshold(
                     gContactBreakingThreshold);
    std::cout << ",\"child_breaking_threshold_bt\":"
              << car->_childHitboxShape.getContactBreakingThreshold(gContactBreakingThreshold);
    std::cout << "}\n";
}

struct CaptureClosestPoint final : btDiscreteCollisionDetectorInterface::Result {
    bool found = false;
    btVector3 normal = btVector3(0, 0, 0);
    btVector3 point = btVector3(0, 0, 0);
    btScalar distance = 0;

    void setShapeIdentifiersA(int, int) override {}
    void setShapeIdentifiersB(int, int) override {}
    void addContactPoint(
        const btVector3& normalOnBInWorld,
        const btVector3& pointInWorld,
        btScalar depth) override {
        found = true;
        normal = normalOnBInWorld;
        point = pointInWorld;
        distance = depth;
    }
};

void PrintGjkProbe(
    Car* car,
    Arena* arena,
    int worldBodyIndex,
    int faceIndex,
    int tick) {
    const btRigidBody* worldBody = arena->_worldCollisionRBs[worldBodyIndex];
    const auto* bvh = dynamic_cast<const btBvhTriangleMeshShape*>(
        worldBody->getCollisionShape());
    if (!bvh) {
        return;
    }
    const unsigned char* vertexBase = nullptr;
    const unsigned char* indexBase = nullptr;
    int vertices = 0;
    int faces = 0;
    int vertexStride = 0;
    int indexStride = 0;
    bvh->getMeshInterface()->getLockedReadOnlyVertexIndexBase(
        &vertexBase, vertices, vertexStride, &indexBase, indexStride, faces);
    const auto* indices = reinterpret_cast<const int*>(
        indexBase + faceIndex * indexStride);
    btVector3 triangle[3];
    for (int corner = 0; corner < 3; ++corner) {
        const auto* values = reinterpret_cast<const btScalar*>(
            vertexBase + indices[corner] * vertexStride);
        triangle[corner] = btVector3(values[0], values[1], values[2]);
    }
    bvh->getMeshInterface()->unLockReadOnlyVertexBase(0);

    btTriangleShape triangleShape(triangle[0], triangle[1], triangle[2]);
    triangleShape.setMargin(bvh->getMargin());
    btVoronoiSimplexSolver simplex;
    btGjkEpaPenetrationDepthSolver penetrationDepthSolver;
    btGjkPairDetector detector(
        &car->_childHitboxShape,
        &triangleShape,
        &simplex,
        &penetrationDepthSolver);
    btGjkPairDetector::ClosestPointInput input;
    const btScalar maximumDistance = car->_childHitboxShape.getMargin()
        + triangleShape.getMargin()
        + car->_compoundShape.getContactBreakingThreshold(gContactBreakingThreshold);
    input.m_maximumDistanceSquared = maximumDistance * maximumDistance;
    input.m_transformA = car->_rigidBody.getWorldTransform()
        * car->_compoundShape.getChildTransform(0);
    input.m_transformB = worldBody->getWorldTransform();
    CaptureClosestPoint result;
    detector.getClosestPoints(input, result);

    btVoronoiSimplexSolver traceSimplex;
    traceSimplex.reset();
    btTransform localA = input.m_transformA;
    btTransform localB = input.m_transformB;
    const btVector3 positionOffset =
        (localA.getOrigin() + localB.getOrigin()) * btScalar(0.5);
    localA.getOrigin() -= positionOffset;
    localB.getOrigin() -= positionOffset;
    btVector3 traceAxis(0, 1, 0);
    btScalar squaredDistance = BT_LARGE_FLOAT;
    constexpr btScalar traceRelativeError2 = btScalar(1.0e-6);
    int traceCurIter = 0;
    std::cout << "{\"record\":\"gjk_iterations\",\"tick\":" << tick
              << ",\"world_body_index\":" << worldBodyIndex
              << ",\"face\":" << faceIndex
              << ",\"transform_a_origin\":";
    PrintVec(input.m_transformA.getOrigin());
    std::cout << ",\"transform_a_basis\":[";
    PrintVec(input.m_transformA.getBasis()[0]);
    std::cout << ',';
    PrintVec(input.m_transformA.getBasis()[1]);
    std::cout << ',';
    PrintVec(input.m_transformA.getBasis()[2]);
    std::cout << "]"
              << ",\"position_offset\":";
    PrintVec(positionOffset);
    std::cout << ",\"steps\":[";
    for (int iteration = 0; iteration < 1002; ++iteration) {
        const btVector3 directionA = (-traceAxis) * input.m_transformA.getBasis();
        const btVector3 directionB = traceAxis * input.m_transformB.getBasis();
        const btVector3 localP =
            car->_childHitboxShape.localGetSupportVertexWithoutMarginNonVirtual(
                directionA);
        const btVector3 localQ =
            triangleShape.localGetSupportVertexWithoutMarginNonVirtual(directionB);
        const btVector3 p = localA(localP);
        const btVector3 q = localB(localQ);
        const btVector3 w = p - q;
        const btScalar delta = traceAxis.dot(w);
        btVector3 priorP[VORONOI_SIMPLEX_MAX_VERTS];
        btVector3 priorQ[VORONOI_SIMPLEX_MAX_VERTS];
        btVector3 priorW[VORONOI_SIMPLEX_MAX_VERTS];
        const int priorCount = traceSimplex.getSimplex(priorP, priorQ, priorW);
        btScalar nearestPrior = BT_LARGE_FLOAT;
        for (int index = 0; index < priorCount; ++index) {
            nearestPrior = btMin(nearestPrior, priorW[index].distance2(w));
        }
        const bool repeated = traceSimplex.inSimplex(w);
        const bool maximumDistanceExit =
            delta > btScalar(0.0)
            && delta * delta > squaredDistance * input.m_maximumDistanceSquared;
        const btScalar f0 = squaredDistance - delta;
        const btScalar f1 = squaredDistance * traceRelativeError2;
        if (iteration) {
            std::cout << ',';
        }
        std::cout << "{\"iteration\":" << iteration
                  << ",\"axis\":";
        PrintVec(traceAxis);
        std::cout << ",\"direction_a\":";
        PrintVec(directionA);
        std::cout << ",\"local_p\":";
        PrintVec(localP);
        std::cout << ",\"local_q\":";
        PrintVec(localQ);
        std::cout << ",\"p\":";
        PrintVec(p);
        std::cout << ",\"q\":";
        PrintVec(q);
        std::cout << ",\"w\":";
        PrintVec(w);
        std::cout << ",\"delta\":" << delta
                  << ",\"squared_distance_before\":" << squaredDistance
                  << ",\"maximum_distance_squared\":"
                  << input.m_maximumDistanceSquared
                  << ",\"maximum_distance_exit\":"
                  << (maximumDistanceExit ? "true" : "false")
                  << ",\"nearest_prior_distance_sq\":" << nearestPrior
                  << ",\"repeated\":" << (repeated ? "true" : "false")
                  << ",\"f0\":" << f0 << ",\"f1\":" << f1;
        if (maximumDistanceExit) {
            std::cout << ",\"exit_reason\":\"maximum_distance\"}";
            break;
        }
        if (repeated) {
            std::cout << ",\"exit_reason\":\"in_simplex\"}";
            break;
        }
        if (f0 <= f1) {
            std::cout << ",\"exit_reason\":\"relative_progress\"}";
            break;
        }
        traceSimplex.addVertex(w, p, q);
        btVector3 nextAxis;
        const bool closest = traceSimplex.closest(nextAxis);
        std::cout << ",\"closest\":" << (closest ? "true" : "false")
                  << ",\"next_axis\":";
        PrintVec(nextAxis);
        std::cout << ",\"squared_distance_after\":" << nextAxis.length2()
                  << ",\"simplex_count_after\":" << traceSimplex.numVertices()
                  << ",\"cached_point_a\":";
        PrintVec(traceSimplex.m_cachedP1);
        std::cout << ",\"cached_point_b\":";
        PrintVec(traceSimplex.m_cachedP2);
        std::cout << ",\"cached_v\":";
        PrintVec(traceSimplex.m_cachedV);
        std::cout << ",\"cached_valid\":"
                  << (traceSimplex.m_cachedValidClosest ? "true" : "false")
                  << ",\"cached_degenerate\":"
                  << (traceSimplex.m_cachedBC.m_degenerate ? "true" : "false")
                  << ",\"used_vertices\":["
                  << (traceSimplex.m_cachedBC.m_usedVertices.usedVertexA ? "true" : "false")
                  << ','
                  << (traceSimplex.m_cachedBC.m_usedVertices.usedVertexB ? "true" : "false")
                  << ','
                  << (traceSimplex.m_cachedBC.m_usedVertices.usedVertexC ? "true" : "false")
                  << ','
                  << (traceSimplex.m_cachedBC.m_usedVertices.usedVertexD ? "true" : "false")
                  << "],\"barycentric\":["
                  << traceSimplex.m_cachedBC.m_barycentricCoords[0] << ','
                  << traceSimplex.m_cachedBC.m_barycentricCoords[1] << ','
                  << traceSimplex.m_cachedBC.m_barycentricCoords[2] << ','
                  << traceSimplex.m_cachedBC.m_barycentricCoords[3] << ']';
        if (!closest) {
            std::cout << ",\"exit_reason\":\"closest_failed\"}";
            break;
        }
        if (nextAxis.length2() < traceRelativeError2) {
            std::cout << ",\"exit_reason\":\"axis_near_zero\"}";
            break;
        }
        const btScalar previousSquaredDistance = squaredDistance;
        squaredDistance = nextAxis.length2();
        if (previousSquaredDistance - squaredDistance
            <= SIMD_EPSILON * previousSquaredDistance) {
            std::cout << ",\"exit_reason\":\"insufficient_progress\"}";
            break;
        }
        traceAxis = nextAxis;
        if (traceCurIter++ > 1000) {
            std::cout << ",\"exit_reason\":\"iteration_limit\"}";
            break;
        }
        if (traceSimplex.fullSimplex()) {
            std::cout << ",\"exit_reason\":\"full_simplex\"}";
            break;
        }
        std::cout << ",\"exit_reason\":null}";
    }
    std::cout << "]}\n";

    std::cout << "{\"record\":\"gjk_probe\",\"tick\":" << tick
              << ",\"world_body_index\":" << worldBodyIndex
              << ",\"face\":" << faceIndex
              << ",\"margin_a\":" << detector.m_marginA
              << ",\"margin_b\":" << detector.m_marginB
              << ",\"maximum_distance\":" << maximumDistance
              << ",\"iterations\":" << detector.m_curIter
              << ",\"degenerate\":" << detector.m_degenerateSimplex
              << ",\"method\":" << detector.m_lastUsedMethod
              << ",\"cached_distance\":" << detector.m_cachedSeparatingDistance
              << ",\"axis\":";
    PrintVec(detector.m_cachedSeparatingAxis);
    std::cout << ",\"found\":" << (result.found ? "true" : "false")
              << ",\"normal\":";
    PrintVec(result.normal);
    std::cout << ",\"point_b\":";
    PrintVec(result.point);
    std::cout << ",\"distance\":" << result.distance
              << ",\"equal_vertex_threshold\":"
              << simplex.getEqualVertexThreshold();
    btVector3 simplexP[VORONOI_SIMPLEX_MAX_VERTS];
    btVector3 simplexQ[VORONOI_SIMPLEX_MAX_VERTS];
    btVector3 simplexW[VORONOI_SIMPLEX_MAX_VERTS];
    const int simplexCount = simplex.getSimplex(simplexP, simplexQ, simplexW);
    std::cout << ",\"simplex\":[";
    for (int index = 0; index < simplexCount; ++index) {
        if (index) {
            std::cout << ',';
        }
        std::cout << "{\"w\":";
        PrintVec(simplexW[index]);
        std::cout << ",\"p\":";
        PrintVec(simplexP[index]);
        std::cout << ",\"q\":";
        PrintVec(simplexQ[index]);
        std::cout << '}';
    }
    std::cout << "]}\n";
}

void PrintEpaProbe(
    Car* car,
    Arena* arena,
    int worldBodyIndex,
    int faceIndex,
    int tick) {
    const btRigidBody* worldBody = arena->_worldCollisionRBs[worldBodyIndex];
    const auto* bvh = dynamic_cast<const btBvhTriangleMeshShape*>(
        worldBody->getCollisionShape());
    if (!bvh) {
        return;
    }
    const unsigned char* vertexBase = nullptr;
    const unsigned char* indexBase = nullptr;
    int vertices = 0;
    int faces = 0;
    int vertexStride = 0;
    int indexStride = 0;
    bvh->getMeshInterface()->getLockedReadOnlyVertexIndexBase(
        &vertexBase, vertices, vertexStride, &indexBase, indexStride, faces);
    const auto* indices = reinterpret_cast<const int*>(
        indexBase + faceIndex * indexStride);
    btVector3 triangle[3];
    for (int corner = 0; corner < 3; ++corner) {
        const auto* values = reinterpret_cast<const btScalar*>(
            vertexBase + indices[corner] * vertexStride);
        triangle[corner] = btVector3(values[0], values[1], values[2]);
    }
    bvh->getMeshInterface()->unLockReadOnlyVertexBase(0);

    btTriangleShape triangleShape(triangle[0], triangle[1], triangle[2]);
    triangleShape.setMargin(bvh->getMargin());
    const btTransform transformA = car->_rigidBody.getWorldTransform()
        * car->_compoundShape.getChildTransform(0);
    const btTransform transformB = worldBody->getWorldTransform();
    const btVector3 guess = (transformB.getOrigin() - transformA.getOrigin()).safeNormalize();
    btGjkEpaSolver2::sResults result{};
    const bool found = btGjkEpaSolver2::Penetration(
        &car->_childHitboxShape, transformA, &triangleShape, transformB,
        guess, result);
    std::cout << "{\"record\":\"epa_probe\",\"tick\":" << tick
              << ",\"world_body_index\":" << worldBodyIndex
              << ",\"face\":" << faceIndex
              << ",\"found\":" << (found ? "true" : "false")
              << ",\"status\":" << static_cast<int>(result.status)
              << ",\"normal\":";
    PrintVec(result.normal);
    std::cout << ",\"distance\":" << result.distance
              << ",\"witness_a\":";
    PrintVec(result.witnesses[0]);
    std::cout << ",\"witness_b\":";
    PrintVec(result.witnesses[1]);
    std::cout << "}\n";

    const btVector3 guesses[] = {
        btVector3(transformB.getOrigin() - transformA.getOrigin()).safeNormalize(),
        btVector3(transformA.getOrigin() - transformB.getOrigin()).safeNormalize(),
        btVector3(0, 0, 1),
        btVector3(0, 1, 0),
        btVector3(1, 0, 0),
        btVector3(1, 1, 0),
        btVector3(1, 1, 1),
        btVector3(0, 1, 1),
        btVector3(1, 0, 1),
    };
    btVoronoiSimplexSolver solverSimplex;
    btVector3 selectedNormal(0, 0, 0);
    btVector3 selectedWitnessA(0, 0, 0);
    btVector3 selectedWitnessB(0, 0, 0);
    int selectedAttempt = -1;
    const char* selectedMode = "none";
    bool solverReturn = false;
    std::cout << "{\"record\":\"epa_solver_trace\",\"tick\":" << tick
              << ",\"world_body_index\":" << worldBodyIndex
              << ",\"face\":" << faceIndex << ",\"attempts\":[";
    for (int attempt = 0; attempt < 9; ++attempt) {
        if (attempt) {
            std::cout << ',';
        }
        solverSimplex.reset();
        btGjkEpaSolver2::sResults penetrationResult{};
        const bool penetrationFound = btGjkEpaSolver2::Penetration(
            &car->_childHitboxShape,
            transformA,
            &triangleShape,
            transformB,
            guesses[attempt],
            penetrationResult);
        std::cout << "{\"attempt\":" << attempt << ",\"guess\":";
        PrintVec(guesses[attempt]);
        std::cout << ",\"penetration\":{\"found\":"
                  << (penetrationFound ? "true" : "false")
                  << ",\"status\":" << static_cast<int>(penetrationResult.status)
                  << ",\"normal\":";
        PrintVec(penetrationResult.normal);
        std::cout << ",\"distance\":" << penetrationResult.distance
                  << ",\"witness_a\":";
        PrintVec(penetrationResult.witnesses[0]);
        std::cout << ",\"witness_b\":";
        PrintVec(penetrationResult.witnesses[1]);
        std::cout << '}';
        if (penetrationFound) {
            selectedNormal = penetrationResult.normal;
            selectedWitnessA = penetrationResult.witnesses[0];
            selectedWitnessB = penetrationResult.witnesses[1];
            selectedAttempt = attempt;
            selectedMode = "penetration";
            solverReturn = true;
            std::cout << ",\"distance_fallback\":null,\"selected\":true}";
            break;
        }

        btGjkEpaSolver2::sResults distanceResult{};
        const bool distanceFound = btGjkEpaSolver2::Distance(
            &car->_childHitboxShape,
            transformA,
            &triangleShape,
            transformB,
            guesses[attempt],
            distanceResult);
        std::cout << ",\"distance_fallback\":{\"found\":"
                  << (distanceFound ? "true" : "false")
                  << ",\"status\":" << static_cast<int>(distanceResult.status)
                  << ",\"normal\":";
        PrintVec(distanceResult.normal);
        std::cout << ",\"distance\":" << distanceResult.distance
                  << ",\"witness_a\":";
        PrintVec(distanceResult.witnesses[0]);
        std::cout << ",\"witness_b\":";
        PrintVec(distanceResult.witnesses[1]);
        std::cout << "},\"selected\":" << (distanceFound ? "true" : "false")
                  << '}';
        if (distanceFound) {
            selectedNormal = distanceResult.normal;
            selectedWitnessA = distanceResult.witnesses[0];
            selectedWitnessB = distanceResult.witnesses[1];
            selectedAttempt = attempt;
            selectedMode = "distance";
            break;
        }
    }
    std::cout << "],\"solver_return\":" << (solverReturn ? "true" : "false")
              << ",\"selected_attempt\":" << selectedAttempt
              << ",\"selected_mode\":\"" << selectedMode
              << "\",\"selected_normal\":";
    PrintVec(selectedNormal);
    std::cout << ",\"selected_witness_a\":";
    PrintVec(selectedWitnessA);
    std::cout << ",\"selected_witness_b\":";
    PrintVec(selectedWitnessB);
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage: rocketsim_diagnostic COLLISION_ROOT SCENARIO TICKS\n";
            return 2;
        }
        const std::string collisionRoot = argv[1];
        const std::string scenarioName = argv[2];
        const bool customProbe = scenarioName == "custom_probe";
        const bool customScenario =
            scenarioName == "custom" || scenarioName == "custom_full"
            || customProbe;
        const int ticks = std::stoi(argv[3]);
        if (ticks < 0 || ticks > 600) {
            throw std::runtime_error("ticks must be in [0, 600]");
        }

        RocketSim::Init(collisionRoot, true);
        ArenaConfig config;
        config.noBallRot = false;
        Arena* arena = Arena::Create(GameMode::SOCCAR, config, 120.0f);
        arena->SetCarCarCollision(false);
        arena->SetCarBallCollision(false);
        Car* car = arena->AddCar(Team::BLUE, CAR_CONFIG_OCTANE);
        TracingVehicleRaycaster tracingVehicleRaycaster(&arena->_bulletWorld);
        tracingVehicleRaycaster.addedFilterMask =
            car->_bulletVehicleRaycaster.addedFilterMask;
        if (customScenario) {
            car->_bulletVehicle.m_vehicleRaycaster = &tracingVehicleRaycaster;
            btSequentialImpulseConstraintSolver* solver =
                arena->_bulletWorld.getConstraintSolver();
            const int solverMode = arena->_bulletWorld.getSolverInfo().m_solverMode;
            solver->setupSolverFunctions((solverMode & SOLVER_SIMD) != 0);
            solver->m_cachedSolverMode = solverMode;
            priorGenericRowSolver = solver->m_resolveSingleConstraintRowGeneric;
            priorLowerLimitRowSolver =
                solver->m_resolveSingleConstraintRowLowerLimit;
            priorSplitRowSolver = solver->m_resolveSplitPenetrationImpulse;
            solver->m_resolveSingleConstraintRowGeneric = &TraceGenericSolverRow;
            solver->m_resolveSingleConstraintRowLowerLimit =
                &TraceLowerLimitSolverRow;
            solver->m_resolveSplitPenetrationImpulse = &TraceSplitSolverRow;
        }
        // Breadth-validation custom cases must use the same isolated one-car
        // authority arenas as the Python oracle.  Even with car/car collision
        // disabled, adding an unrelated car changes Bullet's static-constraint
        // ordering on later ticks.
        Car* remote = customScenario
            ? nullptr
            : arena->AddCar(Team::ORANGE, CAR_CONFIG_OCTANE);
        traceCar = car;
        traceArena = arena;
        priorContactAddedCallback = gContactAddedCallback;
        gContactAddedCallback = &TraceContactAdded;

        const Scenario scenario = customScenario
            ? MakeCustomScenario(argc, argv)
            : MakeScenario(scenarioName);
        CarState state;
        state.pos = scenario.pos;
        state.vel = scenario.vel;
        state.angVel = scenario.angVel;
        state.rotMat = scenario.rotMat;
        state.isOnGround = scenario.onGround;
        state.handbrakeVal = 0.0f;
        state.boost = scenario.boost;
        if (scenario.seedLastControls) {
            state.lastControls = scenario.controls;
        }
        car->SetState(state);
        car->controls = scenario.controls;

        if (remote) {
            CarState remoteState;
            remoteState.pos = Vec(2500, -2500, 17);
            remoteState.isOnGround = true;
            remote->SetState(remoteState);
        }

        BallState ballState;
        ballState.pos = Vec(0, 0, 1500);
        arena->ball->SetState(ballState);

        std::cout << std::setprecision(9);
        PrintShape(car);
        PrintState("initial", 0, car, arena);
        const bool staged = scenarioName == "custom" || customProbe
            || (argc >= 5 && std::string(argv[4]) == "staged");
        for (int tick = 1; tick <= ticks; ++tick) {
            traceTick = tick;
            solverRowCall = 0;
            if (staged) {
                arena->_bulletWorld.setWorldUserInfo(arena);
                const btVector3 preVehicleLinear = car->_rigidBody.getLinearVelocity();
                const btVector3 preVehicleAngular = car->_rigidBody.getAngularVelocity();
                if (customScenario) {
                    PrintAirDampingReplay(tick, car);
                }
                car->_PreTickUpdate(arena->gameMode, arena->tickTime, arena->_mutatorConfig);
                if (remote) {
                    remote->_PreTickUpdate(
                        arena->gameMode, arena->tickTime, arena->_mutatorConfig);
                }
                arena->ball->_PreTickUpdate(arena->gameMode, arena->tickTime);
                if (customScenario) {
                    PrintWheelApplyReplay(
                        car,
                        preVehicleLinear,
                        preVehicleAngular,
                        arena->tickTime,
                        tick);
                }
                PrintState("after_car_pre", tick, car, arena);
                if (scenarioName == "custom" || customProbe) {
                    if (!customProbe) {
                        PrintBvhTraversal(car, arena, tick);
                    }
                    for (const auto& [bodyIndex, faceIndex] : customGjkProbes) {
                        PrintGjkProbe(car, arena, bodyIndex, faceIndex, tick);
                        PrintEpaProbe(car, arena, bodyIndex, faceIndex, tick);
                    }
                }
                if (scenarioName == "ramp_transition" && tick == 563) {
                    PrintGjkProbe(car, arena, 10, 371, tick);
                    PrintGjkProbe(car, arena, 10, 386, tick);
                }
                if (scenarioName == "ramp_transition" && tick == 1) {
                    PrintEpaProbe(car, arena, 8, 744, tick);
                    PrintEpaProbe(car, arena, 8, 693, tick);
                }
                if (scenarioName == "ramp_sim_tick533" && tick == 1) {
                    PrintGjkProbe(car, arena, 10, 371, tick);
                }
                if (scenarioName == "back_wall_transition" && tick == 99) {
                    PrintGjkProbe(car, arena, 10, 831, tick);
                    PrintGjkProbe(car, arena, 10, 825, tick);
                }
                arena->_bulletWorld.stepSimulation(arena->tickTime, 0, arena->tickTime);
                PrintState("after_bullet", tick, car, arena);
                if (scenarioName == "back_wall_transition" && tick == 440) {
                    PrintGjkProbe(car, arena, 11, 632, tick);
                    PrintGjkProbe(car, arena, 11, 634, tick);
                }
                car->_PostTickUpdate(arena->gameMode, arena->tickTime, arena->_mutatorConfig);
                car->_FinishPhysicsTick(arena->_mutatorConfig);
                if (remote) {
                    remote->_PostTickUpdate(
                        arena->gameMode, arena->tickTime, arena->_mutatorConfig);
                    remote->_FinishPhysicsTick(arena->_mutatorConfig);
                }
                arena->ball->_FinishPhysicsTick(arena->_mutatorConfig);
                arena->tickCount++;
            } else {
                arena->Step(1);
            }
            PrintState("post", tick, car, arena);
        }
        gContactAddedCallback = priorContactAddedCallback;
        traceCar = nullptr;
        traceArena = nullptr;
        traceTick = 0;
        delete arena;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
