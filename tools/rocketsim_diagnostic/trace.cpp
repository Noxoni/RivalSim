#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "RocketSim.h"
#include "bullet3-3.24/BulletCollision/CollisionShapes/btConvexPolyhedron.h"
#include "bullet3-3.24/BulletCollision/CollisionShapes/btTriangleShape.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btGjkEpa2.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h"

using namespace RocketSim;

extern btScalar gContactBreakingThreshold;

namespace {

struct Scenario {
    Vec pos;
    Vec vel;
    RotMat rotMat;
    CarControls controls;
    bool onGround;
    float boost;
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
    btVector3 normal;
    btVector3 point;
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

void PrintGjkProbe(Car* car, Arena* arena, int worldBodyIndex, int faceIndex) {
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
    btGjkPairDetector detector(
        &car->_childHitboxShape, &triangleShape, &simplex, nullptr);
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
    std::cout << "{\"record\":\"gjk_iterations\",\"face\":" << faceIndex
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
    for (int iteration = 0; iteration < 16; ++iteration) {
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
                  << ",\"nearest_prior_distance_sq\":" << nearestPrior
                  << ",\"repeated\":" << (repeated ? "true" : "false");
        if (repeated) {
            std::cout << '}';
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
                  << '}';
        if (!closest) {
            break;
        }
        squaredDistance = nextAxis.length2();
        traceAxis = nextAxis;
    }
    std::cout << "]}\n";

    std::cout << "{\"record\":\"gjk_probe\",\"face\":" << faceIndex
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

void PrintEpaProbe(Car* car, Arena* arena, int worldBodyIndex, int faceIndex) {
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
    btGjkEpaSolver2::sResults result;
    const bool found = btGjkEpaSolver2::Penetration(
        &car->_childHitboxShape, transformA, &triangleShape, transformB,
        guess, result);
    std::cout << "{\"record\":\"epa_probe\",\"face\":" << faceIndex
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
        Car* remote = arena->AddCar(Team::ORANGE, CAR_CONFIG_OCTANE);

        const Scenario scenario = MakeScenario(scenarioName);
        CarState state;
        state.pos = scenario.pos;
        state.vel = scenario.vel;
        state.rotMat = scenario.rotMat;
        state.isOnGround = scenario.onGround;
        state.handbrakeVal = 0.0f;
        state.boost = scenario.boost;
        car->SetState(state);
        car->controls = scenario.controls;

        CarState remoteState;
        remoteState.pos = Vec(2500, -2500, 17);
        remoteState.isOnGround = true;
        remote->SetState(remoteState);

        BallState ballState;
        ballState.pos = Vec(0, 0, 1500);
        arena->ball->SetState(ballState);

        std::cout << std::setprecision(9);
        PrintShape(car);
        PrintState("initial", 0, car, arena);
        const bool staged = argc >= 5 && std::string(argv[4]) == "staged";
        for (int tick = 1; tick <= ticks; ++tick) {
            if (staged) {
                arena->_bulletWorld.setWorldUserInfo(arena);
                car->_PreTickUpdate(arena->gameMode, arena->tickTime, arena->_mutatorConfig);
                remote->_PreTickUpdate(arena->gameMode, arena->tickTime, arena->_mutatorConfig);
                arena->ball->_PreTickUpdate(arena->gameMode, arena->tickTime);
                PrintState("after_car_pre", tick, car, arena);
                if (scenarioName == "ramp_transition" && tick == 563) {
                    PrintGjkProbe(car, arena, 10, 371);
                    PrintGjkProbe(car, arena, 10, 386);
                }
                if (scenarioName == "ramp_transition" && tick == 1) {
                    PrintEpaProbe(car, arena, 8, 744);
                    PrintEpaProbe(car, arena, 8, 693);
                }
                if (scenarioName == "ramp_sim_tick533" && tick == 1) {
                    PrintGjkProbe(car, arena, 10, 371);
                }
                if (scenarioName == "back_wall_transition" && tick == 99) {
                    PrintGjkProbe(car, arena, 10, 831);
                    PrintGjkProbe(car, arena, 10, 825);
                }
                arena->_bulletWorld.stepSimulation(arena->tickTime, 0, arena->tickTime);
                PrintState("after_bullet", tick, car, arena);
                if (scenarioName == "back_wall_transition" && tick == 440) {
                    PrintGjkProbe(car, arena, 11, 632);
                    PrintGjkProbe(car, arena, 11, 634);
                }
                car->_PostTickUpdate(arena->gameMode, arena->tickTime, arena->_mutatorConfig);
                car->_FinishPhysicsTick(arena->_mutatorConfig);
                remote->_PostTickUpdate(arena->gameMode, arena->tickTime, arena->_mutatorConfig);
                remote->_FinishPhysicsTick(arena->_mutatorConfig);
                arena->ball->_FinishPhysicsTick(arena->_mutatorConfig);
                arena->tickCount++;
            } else {
                arena->Step(1);
            }
            PrintState("post", tick, car, arena);
        }
        delete arena;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
