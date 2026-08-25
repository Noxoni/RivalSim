#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "RocketSim.h"
#include "bullet3-3.24/BulletCollision/CollisionShapes/btConvexPolyhedron.h"
#include "bullet3-3.24/BulletCollision/CollisionShapes/btSphereShape.h"
#include "bullet3-3.24/BulletCollision/CollisionShapes/btTriangleShape.h"
#include "bullet3-3.24/BulletCollision/BroadphaseCollision/btRSBroadphase.h"
#include "bullet3-3.24/BulletCollision/CollisionDispatch/btManifoldResult.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btGjkEpa2.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h"
#include "bullet3-3.24/BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h"
#include "bullet3-3.24/BulletDynamics/Vehicle/btDefaultVehicleRaycaster.h"
#include "bullet3-3.24/LinearMath/btTransformUtil.h"

using namespace RocketSim;

void dLineClosestApproach(
    const btVector3& pa,
    const btVector3& ua,
    const btVector3& pb,
    const btVector3& ub,
    btScalar* alpha,
    btScalar* beta);

extern btScalar gContactBreakingThreshold;

namespace {

void PrintVec(const btVector3& value);
void PrintExtraHitReplay(int tick, Car* car, Ball* ball, Arena* arena);

void PrintCarPairTransforms(int tick, Car* carA, Car* carB) {
    const btTransform childA = carA->_rigidBody.getWorldTransform()
        * carA->_compoundShape.getChildTransform(0);
    const btTransform childB = carB->_rigidBody.getWorldTransform()
        * carB->_compoundShape.getChildTransform(0);
    std::cout << "{\"record\":\"car_pair_transforms\",\"tick\":" << tick
              << ",\"root_a\":";
    PrintVec(carA->_rigidBody.getWorldTransform().getOrigin());
    std::cout << ",\"child_a\":";
    PrintVec(childA.getOrigin());
    std::cout << ",\"root_b\":";
    PrintVec(carB->_rigidBody.getWorldTransform().getOrigin());
    std::cout << ",\"child_b\":";
    PrintVec(childB.getOrigin());
    std::cout << "}\n";
}

void PrintCarPairBroadphasePrediction(int tick, Car* carA, Car* carB, btScalar dt) {
    const auto bounds = [dt](Car* car, btVector3& minimum, btVector3& maximum) {
        btTransform predicted;
        car->_rigidBody.predictIntegratedTransform(dt, predicted);
        btVector3 predictedMinimum;
        btVector3 predictedMaximum;
        car->_rigidBody.getCollisionShape()->getAabb(
            car->_rigidBody.getWorldTransform(), minimum, maximum);
        car->_rigidBody.getCollisionShape()->getAabb(
            predicted, predictedMinimum, predictedMaximum);
        const btVector3 threshold(
            gContactBreakingThreshold,
            gContactBreakingThreshold,
            gContactBreakingThreshold);
        minimum -= threshold;
        maximum += threshold;
        predictedMinimum -= threshold;
        predictedMaximum += threshold;
        minimum.setMin(predictedMinimum);
        maximum.setMax(predictedMaximum);
    };
    btVector3 minimumA;
    btVector3 maximumA;
    btVector3 minimumB;
    btVector3 maximumB;
    bounds(carA, minimumA, maximumA);
    bounds(carB, minimumB, maximumB);
    std::cout << "{\"record\":\"car_pair_broadphase_prediction\",\"tick\":"
              << tick << ",\"minimum_a\":";
    PrintVec(minimumA);
    std::cout << ",\"maximum_a\":";
    PrintVec(maximumA);
    std::cout << ",\"minimum_b\":";
    PrintVec(minimumB);
    std::cout << ",\"maximum_b\":";
    PrintVec(maximumB);
    std::cout << ",\"gap_a_to_b\":";
    PrintVec(minimumB - maximumA);
    std::cout << ",\"gap_b_to_a\":";
    PrintVec(minimumA - maximumB);
    std::cout << "}\n";
}

void PrintCarPairBroadphaseResident(int tick, Car* carA, Car* carB, Arena* arena) {
    btVector3 minimumA;
    btVector3 maximumA;
    btVector3 minimumB;
    btVector3 maximumB;
    auto* broadphase = arena->_bulletWorld.getBroadphase();
    broadphase->getAabb(carA->_rigidBody.getBroadphaseHandle(), minimumA, maximumA);
    broadphase->getAabb(carB->_rigidBody.getBroadphaseHandle(), minimumB, maximumB);
    const bool hasPair = broadphase->getOverlappingPairCache()->findPair(
        carA->_rigidBody.getBroadphaseHandle(),
        carB->_rigidBody.getBroadphaseHandle()) != nullptr;
    const auto* proxyA = static_cast<const btRSBroadphaseProxy*>(
        carA->_rigidBody.getBroadphaseHandle());
    const auto* proxyB = static_cast<const btRSBroadphaseProxy*>(
        carB->_rigidBody.getBroadphaseHandle());
    std::cout << "{\"record\":\"car_pair_broadphase_resident\",\"tick\":"
              << tick << ",\"minimum_a\":";
    PrintVec(minimumA);
    std::cout << ",\"maximum_a\":";
    PrintVec(maximumA);
    std::cout << ",\"minimum_b\":";
    PrintVec(minimumB);
    std::cout << ",\"maximum_b\":";
    PrintVec(maximumB);
    std::cout << ",\"has_pair\":" << (hasPair ? "true" : "false")
              << ",\"cell_a\":[" << proxyA->iIdx << ',' << proxyA->jIdx
              << ',' << proxyA->kIdx << ']'
              << ",\"cell_b\":[" << proxyB->iIdx << ',' << proxyB->jIdx
              << ',' << proxyB->kIdx << ']'
              << "}\n";
}

void PrintCarPairFaceAxes(int tick, Car* carA, Car* carB) {
    // Nested compound dispatch reaches btBoxBoxDetector as car B / car A.
    const btTransform first = carB->_rigidBody.getWorldTransform()
        * carB->_compoundShape.getChildTransform(0);
    const btTransform second = carA->_rigidBody.getWorldTransform()
        * carA->_compoundShape.getChildTransform(0);
    const btVector3 p = second.getOrigin() - first.getOrigin();
    const btVector3 half = carA->_childHitboxShape.getHalfExtentsWithMargin();
    const auto dot = [](const btVector3& a, const btVector3& b) {
        return (a.x() * b.x() + a.y() * b.y()) + a.z() * b.z();
    };
    btScalar relative[3][3];
    btScalar absolute[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            relative[row][column] = dot(
                first.getBasis().getColumn(row),
                second.getBasis().getColumn(column));
            absolute[row][column] = btFabs(relative[row][column]);
        }
    }
    btScalar values[6];
    for (int axis = 0; axis < 3; ++axis) {
        const btScalar expression = dot(first.getBasis().getColumn(axis), p);
        const btScalar radius = half[axis]
            + half[0] * absolute[axis][0]
            + half[1] * absolute[axis][1]
            + half[2] * absolute[axis][2];
        values[axis] = btFabs(expression) - radius;
    }
    for (int axis = 0; axis < 3; ++axis) {
        const btScalar expression = dot(second.getBasis().getColumn(axis), p);
        const btScalar radius = half[0] * absolute[0][axis]
            + half[1] * absolute[1][axis]
            + half[2] * absolute[2][axis]
            + half[axis];
        values[3 + axis] = btFabs(expression) - radius;
    }
    std::cout << "{\"record\":\"car_pair_face_axes\",\"tick\":" << tick
              << ",\"values\":[";
    for (int index = 0; index < 6; ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << "]}\n";
}

void PrintCarPairEdge13(int tick, Car* carA, Car* carB) {
    // Replay the selected dBoxBox2 edge-13 branch with the source's padded
    // matrix layout and expression order. Nested dispatch presents B first.
    const btTransform first = carB->_rigidBody.getWorldTransform()
        * carB->_compoundShape.getChildTransform(0);
    const btTransform second = carA->_rigidBody.getWorldTransform()
        * carA->_compoundShape.getChildTransform(0);
    btScalar r1[12] = {};
    btScalar r2[12] = {};
    for (int row = 0; row < 3; ++row) {
        r1[4 * row] = first.getBasis()[row].x();
        r1[4 * row + 1] = first.getBasis()[row].y();
        r1[4 * row + 2] = first.getBasis()[row].z();
        r2[4 * row] = second.getBasis()[row].x();
        r2[4 * row + 1] = second.getBasis()[row].y();
        r2[4 * row + 2] = second.getBasis()[row].z();
    }
    const auto dot41 = [](const btScalar* matrix, const btVector3& value) {
        return matrix[0] * value[0] + matrix[4] * value[1]
            + matrix[8] * value[2];
    };
    const auto dot44 = [](const btScalar* a, const btScalar* b) {
        return a[0] * b[0] + a[4] * b[4] + a[8] * b[8];
    };
    const auto dot14 = [](const btVector3& value, const btScalar* matrix) {
        return value[0] * matrix[0] + value[1] * matrix[4]
            + value[2] * matrix[8];
    };
    const btVector3 p = second.getOrigin() - first.getOrigin();
    const btVector3 pp(
        dot41(r1, p), dot41(r1 + 1, p), dot41(r1 + 2, p));
    const btScalar r11 = dot44(r1, r2);
    const btScalar r21 = dot44(r1 + 1, r2);
    const btScalar r32 = dot44(r1 + 2, r2 + 1);
    const btScalar r33 = dot44(r1 + 2, r2 + 2);
    const btVector3 half = carA->_childHitboxShape.getHalfExtentsWithMargin();
    const btScalar expression = pp[1] * r11 - pp[0] * r21;
    const btScalar radius =
        half[0] * (btFabs(r21) + btScalar(1.0e-5f))
        + half[1] * (btFabs(r11) + btScalar(1.0e-5f))
        + half[1] * (btFabs(r33) + btScalar(1.0e-5f))
        + half[2] * (btFabs(r32) + btScalar(1.0e-5f));
    btScalar scaledSeparation = btFabs(expression) - radius;
    const btScalar length = btSqrt(r21 * r21 + r11 * r11);
    scaledSeparation /= length;
    btVector3 normalLocal(-r21 / length, r11 / length, 0);
    btVector3 normal(
        r1[0] * normalLocal[0] + r1[1] * normalLocal[1]
            + r1[2] * normalLocal[2],
        r1[4] * normalLocal[0] + r1[5] * normalLocal[1]
            + r1[6] * normalLocal[2],
        r1[8] * normalLocal[0] + r1[9] * normalLocal[1]
            + r1[10] * normalLocal[2]);
    if (expression < 0) normal = -normal;
    btVector3 pa = first.getOrigin();
    btVector3 pb = second.getOrigin();
    for (int axis = 0; axis < 3; ++axis) {
        const btScalar signA = dot14(normal, r1 + axis) > 0 ? 1.0f : -1.0f;
        const btScalar signB = dot14(normal, r2 + axis) > 0 ? -1.0f : 1.0f;
        for (int component = 0; component < 3; ++component) {
            pa[component] += signA * half[axis] * r1[component * 4 + axis];
            pb[component] += signB * half[axis] * r2[component * 4 + axis];
        }
    }
    btVector3 ua(r1[2], r1[6], r1[10]);
    btVector3 ub(r2[0], r2[4], r2[8]);
    const btVector3 prePa = pa;
    const btVector3 prePb = pb;
    btScalar alpha;
    btScalar beta;
    dLineClosestApproach(pa, ua, pb, ub, &alpha, &beta);
    pa += ua * alpha;
    pb += ub * beta;
    std::cout << "{\"record\":\"car_pair_edge13\",\"tick\":" << tick
              << ",\"expression\":" << expression
              << ",\"radius\":" << radius
              << ",\"length\":" << length
              << ",\"scaled_separation\":" << scaledSeparation
              << ",\"normal\":";
    PrintVec(normal);
    std::cout << ",\"pa\":";
    PrintVec(pa);
    std::cout << ",\"pb\":";
    PrintVec(pb);
    std::cout << ",\"pre_pa\":";
    PrintVec(prePa);
    std::cout << ",\"pre_pb\":";
    PrintVec(prePb);
    std::cout << ",\"ua\":";
    PrintVec(ua);
    std::cout << ",\"ub\":";
    PrintVec(ub);
    std::cout << ",\"alpha\":" << alpha << ",\"beta\":" << beta
              << "}\n";
}

Car* traceCar = nullptr;
Car* traceOtherCar = nullptr;
btRigidBody* traceBody = nullptr;
btRigidBody* traceBallBody = nullptr;
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
    const bool carIsA = traceBody && bodyA.m_originalBody == traceBody;
    const bool carIsB = traceBody && bodyB.m_originalBody == traceBody;
    const bool ballIsA = traceBallBody && bodyA.m_originalBody == traceBallBody;
    const bool ballIsB = traceBallBody && bodyB.m_originalBody == traceBallBody;
    const int call = solverRowCall++;
    if (!carIsA && !carIsB && !ballIsA && !ballIsB) {
        return original(bodyA, bodyB, constraint);
    }
    const bool tracedIsA = carIsA || (!carIsB && ballIsA);
    btSolverBody& carBody = tracedIsA ? bodyA : bodyB;
    btSolverBody& otherBody = tracedIsA ? bodyB : bodyA;
    const btVector3 deltaLinearBefore = carBody.m_deltaLinearVelocity;
    const btVector3 deltaAngularBefore = carBody.m_deltaAngularVelocity;
    const btVector3 pushBefore = carBody.m_pushVelocity;
    const btVector3 turnBefore = carBody.m_turnVelocity;
    const btVector3 otherDeltaLinearBefore = otherBody.m_deltaLinearVelocity;
    const btVector3 otherDeltaAngularBefore = otherBody.m_deltaAngularVelocity;
    const btVector3 otherPushBefore = otherBody.m_pushVelocity;
    const btVector3 otherTurnBefore = otherBody.m_turnVelocity;
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
              << ",\"trace_body\":\""
              << ((carIsA || carIsB) ? "car" : "ball")
              << "\",\"trace_side\":\"" << (tracedIsA ? "a" : "b")
              << "\",\"other_body\":\""
              << (otherBody.m_originalBody == traceBody
                      ? "car"
                      : otherBody.m_originalBody == traceBallBody
                          ? "ball"
                          : "static")
              << "\",\"friction_index\":" << constraint.m_frictionIndex
              << ",\"normal_1\":";
    PrintVec(constraint.m_contactNormal1);
    std::cout << ",\"normal_2\":";
    PrintVec(constraint.m_contactNormal2);
    std::cout << ",\"rel_cross_1\":";
    PrintVec(constraint.m_relpos1CrossNormal);
    std::cout << ",\"rel_cross_2\":";
    PrintVec(constraint.m_relpos2CrossNormal);
    if (constraint.m_originalContactPoint) {
        const auto* contact = static_cast<const btManifoldPoint*>(
            constraint.m_originalContactPoint);
        std::cout << ",\"relative_position_a\":";
        PrintVec(
            contact->getPositionWorldOnA() - bodyA.m_worldTransform.getOrigin());
        std::cout << ",\"relative_position_b\":";
        PrintVec(
            contact->getPositionWorldOnB() - bodyB.m_worldTransform.getOrigin());
    }
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
    std::cout << ",\"other_base_linear\":";
    PrintVec(otherBody.m_linearVelocity);
    std::cout << ",\"other_base_angular\":";
    PrintVec(otherBody.m_angularVelocity);
    std::cout << ",\"other_external_force_impulse\":";
    PrintVec(otherBody.m_externalForceImpulse);
    std::cout << ",\"other_external_torque_impulse\":";
    PrintVec(otherBody.m_externalTorqueImpulse);
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
    std::cout << ",\"other_delta_linear_before\":";
    PrintVec(otherDeltaLinearBefore);
    std::cout << ",\"other_delta_linear_after\":";
    PrintVec(otherBody.m_deltaLinearVelocity);
    std::cout << ",\"other_delta_angular_before\":";
    PrintVec(otherDeltaAngularBefore);
    std::cout << ",\"other_delta_angular_after\":";
    PrintVec(otherBody.m_deltaAngularVelocity);
    std::cout << ",\"other_push_velocity_before\":";
    PrintVec(otherPushBefore);
    std::cout << ",\"other_push_velocity_after\":";
    PrintVec(otherBody.m_pushVelocity);
    std::cout << ",\"other_turn_velocity_before\":";
    PrintVec(otherTurnBefore);
    std::cout << ",\"other_turn_velocity_after\":";
    PrintVec(otherBody.m_turnVelocity);
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

void TracePointSphereCast(
    const btVector3& from,
    const btVector3& to,
    const btTransform& sphereTransform,
    const btConvexShape* sphereShape,
    int wheel,
    const char* targetKind) {
    btSphereShape pointShape(btScalar(0));
    pointShape.setMargin(btScalar(0));
    btVoronoiSimplexSolver simplex;
    simplex.reset();
    const btTransform fromA(btQuaternion::getIdentity(), from);
    const btTransform toA(btQuaternion::getIdentity(), to);
    const btVector3 r = to - from;
    btTransform interpolatedA = fromA;
    const btVector3 initialA = fromA(
        pointShape.localGetSupportingVertex(-r * fromA.getBasis()));
    const btVector3 initialB = sphereTransform(
        sphereShape->localGetSupportingVertex(r * sphereTransform.getBasis()));
    btVector3 v = initialA - initialB;
    btVector3 n(0, 0, 0);
    btScalar lambda = 0;
    btScalar dist2 = v.length2();
    int remaining = 32;
    int iteration = 0;
    std::cout << "{\"record\":\"ray_sphere_cast\",\"tick\":" << traceTick
              << ",\"wheel\":" << wheel << ",\"target\":\"" << targetKind
              << "\",\"stage\":\"initial\""
              << ",\"sphere_origin\":";
    PrintVec(sphereTransform.getOrigin());
    std::cout << ",\"sphere_basis\":[";
    PrintVec(sphereTransform.getBasis()[0]);
    std::cout << ',';
    PrintVec(sphereTransform.getBasis()[1]);
    std::cout << ',';
    PrintVec(sphereTransform.getBasis()[2]);
    std::cout << "],\"r\":";
    PrintVec(r);
    std::cout << ",\"support_a\":";
    PrintVec(initialA);
    std::cout << ",\"support_b\":";
    PrintVec(initialB);
    std::cout << ",\"v\":";
    PrintVec(v);
    std::cout << ",\"dist2\":" << dist2 << "}\n";
    while (dist2 > btScalar(0.0001) && remaining--) {
        const btVector3 supportA = interpolatedA(
            pointShape.localGetSupportingVertex(-v * interpolatedA.getBasis()));
        const btVector3 sphereLocalDirection = v * sphereTransform.getBasis();
        btVector3 sphereLocalNormalized = sphereLocalDirection;
        sphereLocalNormalized.normalize();
        const btVector3 supportB = sphereTransform(
            sphereShape->localGetSupportingVertex(v * sphereTransform.getBasis()));
        btVector3 w = supportA - supportB;
        const btScalar vDotW = v.dot(w);
        const btScalar vDotR = v.dot(r);
        const btScalar lambdaBefore = lambda;
        if (lambda > btScalar(1)) break;
        if (vDotW > btScalar(0)) {
            if (vDotR >= -(SIMD_EPSILON * SIMD_EPSILON)) break;
            lambda = lambda - vDotW / vDotR;
            interpolatedA.getOrigin().setInterpolate3(from, to, lambda);
            w = supportA - supportB;
            n = v;
        }
        const bool duplicate = simplex.inSimplex(w);
        if (!duplicate) simplex.addVertex(w, supportA, supportB);
        const bool closest = simplex.closest(v);
        dist2 = closest ? v.length2() : btScalar(0);
        std::cout << "{\"record\":\"ray_sphere_cast\",\"tick\":" << traceTick
                  << ",\"wheel\":" << wheel << ",\"target\":\"" << targetKind
                  << "\",\"stage\":\"iteration\""
                  << ",\"iteration\":" << iteration++
                  << ",\"lambda_before\":" << lambdaBefore
                  << ",\"lambda_after\":" << lambda
                  << ",\"support_a\":";
        PrintVec(supportA);
        std::cout << ",\"support_b\":";
        PrintVec(supportB);
        std::cout << ",\"sphere_local_direction\":";
        PrintVec(sphereLocalDirection);
        std::cout << ",\"sphere_local_length2\":"
                  << sphereLocalDirection.length2();
        std::cout << ",\"sphere_local_normalized\":";
        PrintVec(sphereLocalNormalized);
        std::cout << ",\"w\":";
        PrintVec(w);
        std::cout << ",\"v_dot_w\":" << vDotW
                  << ",\"v_dot_r\":" << vDotR
                  << ",\"duplicate\":" << (duplicate ? "true" : "false")
                  << ",\"simplex_count\":" << simplex.numVertices()
                  << ",\"closest\":" << (closest ? "true" : "false")
                  << ",\"v\":";
        PrintVec(v);
        std::cout << ",\"n\":";
        PrintVec(n);
        std::cout << ",\"dist2\":" << dist2 << "}\n";
    }
    // Re-enter the compiled pinned implementation with an exact iteration
    // budget. This exposes each native advancement without instrumenting (and
    // thereby perturbing) btSubsimplexConvexCast::calcTimeOfImpact itself.
    for (int limit = 1; limit <= MAX_CONVEX_CAST_ITERATIONS; ++limit) {
        btVoronoiSimplexSolver boundedSimplex;
        btSubsimplexConvexCast boundedCast(
            &pointShape, sphereShape, &boundedSimplex);
        btConvexCast::CastResult boundedResult;
        boundedResult.m_subSimplexCastMaxIterations = limit;
        boundedResult.m_subSimplexCastEpsilon = btScalar(0);
        const bool boundedHit = boundedCast.calcTimeOfImpact(
            fromA, toA, sphereTransform, sphereTransform, boundedResult);
        btVector3 boundedPointA;
        btVector3 boundedPointB;
        boundedSimplex.compute_points(boundedPointA, boundedPointB);
        btVector3 boundedCachedV;
        boundedSimplex.backup_closest(boundedCachedV);
        btVector3 boundedP[VORONOI_SIMPLEX_MAX_VERTS];
        btVector3 boundedQ[VORONOI_SIMPLEX_MAX_VERTS];
        btVector3 boundedW[VORONOI_SIMPLEX_MAX_VERTS];
        const int boundedCount = boundedSimplex.getSimplex(
            boundedP, boundedQ, boundedW);
        std::cout << "{\"record\":\"ray_sphere_native_bounded\",\"tick\":"
                  << traceTick << ",\"wheel\":" << wheel
                  << ",\"target\":\"" << targetKind << '\"'
                  << ",\"iterations\":" << limit
                  << ",\"hit\":" << (boundedHit ? "true" : "false")
                  << ",\"fraction\":" << boundedResult.m_fraction
                  << ",\"normal\":";
        PrintVec(boundedResult.m_normal);
        std::cout << ",\"closest_a\":";
        PrintVec(boundedPointA);
        std::cout << ",\"closest_b\":";
        PrintVec(boundedPointB);
        std::cout << ",\"closest_v\":";
        PrintVec(boundedPointA - boundedPointB);
        std::cout << ",\"cached_v\":";
        PrintVec(boundedCachedV);
        std::cout << ",\"simplex_count\":" << boundedCount
                  << ",\"simplex_p\":[";
        for (int vertex = 0; vertex < boundedCount; ++vertex) {
            if (vertex) std::cout << ',';
            PrintVec(boundedP[vertex]);
        }
        std::cout << "],\"simplex_q\":[";
        for (int vertex = 0; vertex < boundedCount; ++vertex) {
            if (vertex) std::cout << ',';
            PrintVec(boundedQ[vertex]);
        }
        std::cout << "],\"simplex_w\":[";
        for (int vertex = 0; vertex < boundedCount; ++vertex) {
            if (vertex) std::cout << ',';
            PrintVec(boundedW[vertex]);
        }
        std::cout << ']';
        std::cout << "}\n";
    }
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
        std::vector<const char*> dynamicCandidates;
        if (traceArena) {
            auto* broadphase = static_cast<btRSBroadphase*>(
                traceArena->_bulletWorld.getBroadphase());
            const auto& cell = broadphase->cells[broadphase->GetCellIdx(from)];
            for (const auto* proxy : cell.dynHandles) {
                const auto* candidate = static_cast<const btCollisionObject*>(
                    proxy->m_clientObject);
                dynamicCandidates.push_back(
                    traceCar && candidate == &traceCar->_rigidBody
                        ? "car_a"
                        : traceOtherCar
                              && candidate == &traceOtherCar->_rigidBody
                            ? "car_b"
                            : candidate == &traceArena->ball->_rigidBody
                                  ? "ball"
                                  : "other");
            }
        }
        if (traceOtherCar && traceTick <= 3) {
            const btTransform childTransform =
                traceOtherCar->_rigidBody.getWorldTransform()
                * traceOtherCar->_compoundShape.getChildTransform(0);
            TracePointSphereCast(
                from,
                to,
                childTransform,
                &traceOtherCar->_childHitboxShape,
                wheel,
                "car_b");
        }
        void* object = btDefaultVehicleRaycaster::castRay(
            from, to, ignoreObject, result);
        if (object && traceArena
            && object == &traceArena->ball->_rigidBody
            && traceTick <= 3) {
            TracePointSphereCast(
                from,
                to,
                traceArena->ball->_rigidBody.getWorldTransform(),
                static_cast<const btConvexShape*>(
                    traceArena->ball->_rigidBody.getCollisionShape()),
                wheel,
                "ball");
        }
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
        std::cout << ",\"dynamic_candidates\":[";
        for (int index = 0; index < dynamicCandidates.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << '\"' << dynamicCandidates[index] << '\"';
        }
        std::cout << ']';
        std::cout << ",\"hit\":" << (object ? "true" : "false");
        if (object) {
            std::cout << ",\"hit_body\":\""
                      << (traceOtherCar
                              && object == &traceOtherCar->_rigidBody
                              ? "car_b"
                              : object == &traceArena->ball->_rigidBody
                                  ? "ball"
                                  : "static")
                      << "\"";
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
    const bool carBallPair = traceCar && traceBallBody && traceArena
        && ((objectA->getCollisionObject() == traceBody
             && objectB->getCollisionObject() == traceBallBody)
            || (objectB->getCollisionObject() == traceBody
                && objectA->getCollisionObject() == traceBallBody));
    if (carBallPair && !traceOtherCar) {
        PrintExtraHitReplay(traceTick, traceCar, traceArena->ball, traceArena);
    }
    const bool involvesTraceCar = traceBody
        && (objectA->getCollisionObject() == traceBody
            || objectB->getCollisionObject() == traceBody);
    if (!involvesTraceCar || !traceArena) {
        return result;
    }
    const btCollisionObject* worldBody =
        objectA->getCollisionObject() == traceBody
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
              << ",\"object_a\":\""
              << (objectA->getCollisionObject() == traceBody
                      ? "car_a"
                      : objectA->getCollisionObject() == traceBallBody
                          ? "car_b"
                          : "static")
              << "\",\"object_b\":\""
              << (objectB->getCollisionObject() == traceBody
                      ? "car_a"
                      : objectB->getCollisionObject() == traceBallBody
                          ? "car_b"
                          : "static")
              << "\""
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
            (manifold->getBody0() == traceBody
             && manifold->getBody1() == worldBody)
            || (manifold->getBody1() == traceBody
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

BallState MakeCustomBallState(int argc, char** argv) {
    constexpr int BALL_VALUE_COUNT = 9;
    constexpr int BALL_BASE_ARGC = 4 + BALL_VALUE_COUNT;
    if (argc != BALL_BASE_ARGC) {
        throw std::runtime_error(
            "ball_custom requires 9 values: pos3 vel3 ang_vel3");
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
    BallState result;
    result.pos = vector();
    result.vel = vector();
    result.angVel = vector();
    result.rotMat = RotMat(
        Vec(1.0f, 0.0f, 0.0f),
        Vec(0.0f, 1.0f, 0.0f),
        Vec(0.0f, 0.0f, 1.0f));
    return result;
}

BallState MakeCarBallCustomBallState(int argc, char** argv) {
    constexpr int CAR_VALUE_COUNT = 21;
    constexpr int BALL_VALUE_COUNT = 9;
    constexpr int EXPECTED_ARGC = 4 + CAR_VALUE_COUNT + BALL_VALUE_COUNT;
    if (argc != EXPECTED_ARGC) {
        throw std::runtime_error(
            "car_ball_custom requires 30 values: car pos3 vel3 ang_vel3 "
            "forward3 right3 up3 throttle steer handbrake, then ball "
            "pos3 vel3 ang_vel3");
    }
    int index = 4 + CAR_VALUE_COUNT;
    const auto vector = [&]() {
        const Vec result(
            std::stof(argv[index]),
            std::stof(argv[index + 1]),
            std::stof(argv[index + 2]));
        index += 3;
        return result;
    };
    BallState result;
    result.pos = vector();
    result.vel = vector();
    result.angVel = vector();
    result.rotMat = RotMat(
        Vec(1.0f, 0.0f, 0.0f),
        Vec(0.0f, 1.0f, 0.0f),
        Vec(0.0f, 0.0f, 1.0f));
    return result;
}

Scenario MakeCarCarCustomScenario(int argc, char** argv, int carIndex) {
    constexpr int CAR_VALUE_COUNT = 22;
    constexpr int EXPECTED_ARGC = 4 + CAR_VALUE_COUNT * 2;
    if (argc != EXPECTED_ARGC) {
        throw std::runtime_error(
            "car_car_custom requires 44 values: for each car, pos3 vel3 "
            "ang_vel3 forward3 right3 up3 throttle steer handbrake "
            "on_ground");
    }
    int index = 4 + carIndex * CAR_VALUE_COUNT;
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
    result.onGround = std::stoi(argv[index++]) != 0;
    result.boost = 100.0f;
    result.seedLastControls = true;
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
    const char* carLabel,
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
              << ",\"car\":\"" << carLabel << "\""
              << ",\"initial_linear_bt\":";
    PrintVec(initialLinear);
    const btScalar sourceForwardSpeedBt =
        initialLinear.dot(car->_bulletVehicle.getForwardVector());
    std::cout << ",\"source_forward_speed_bt\":" << sourceForwardSpeedBt
              << ",\"source_forward_speed_uu\":"
              << sourceForwardSpeedBt * BT_TO_UU;
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
    std::cout << ",\"rigid_basis\":";
    PrintMatrix(car->_rigidBody.getWorldTransform().getBasis());
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

void PrintBallState(const char* phase, int tick, Ball* ball, Arena* arena) {
    const BallState state = ball->GetState();
    const btRigidBody& body = ball->_rigidBody;
    std::cout << "{\"record\":\"ball_state\",\"phase\":\"" << phase
              << "\",\"tick\":" << tick << ",\"pos\":";
    PrintVec(state.pos);
    std::cout << ",\"vel\":";
    PrintVec(state.vel);
    std::cout << ",\"ang_vel\":";
    PrintVec(state.angVel);
    std::cout << ",\"rigid_pos_bt\":";
    PrintVec(body.getWorldTransform().getOrigin());
    std::cout << ",\"rigid_quat\":";
    PrintQuat(body.getWorldTransform().getRotation());
    std::cout << ",\"rigid_basis\":";
    PrintMatrix(body.getWorldTransform().getBasis());
    std::cout << ",\"rigid_vel_bt\":";
    PrintVec(body.getLinearVelocity());
    std::cout << ",\"rigid_ang_vel\":";
    PrintVec(body.getAngularVelocity());
    std::cout << ",\"total_force_bt\":";
    PrintVec(body.getTotalForce());
    std::cout << ",\"total_torque_bt\":";
    PrintVec(body.getTotalTorque());
    std::cout << ",\"inv_inertia_world\":";
    PrintMatrix(body.getInvInertiaTensorWorld());
    std::cout << ",\"manifolds\":[";

    auto* dispatcher = arena->_bulletWorld.getDispatcher();
    bool firstPoint = true;
    for (int manifoldIndex = 0; manifoldIndex < dispatcher->getNumManifolds(); ++manifoldIndex) {
        btPersistentManifold* manifold = dispatcher->getManifoldByIndexInternal(manifoldIndex);
        const bool involvesBall = manifold->getBody0() == &body
            || manifold->getBody1() == &body;
        if (!involvesBall) {
            continue;
        }
        const btCollisionObject* worldBody = manifold->getBody0() == &body
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
            std::cout << "{\"manifold_index\":" << manifoldIndex
                      << ",\"point_index\":" << pointIndex
                      << ",\"world_body_index\":" << worldBodyIndex
                      << ",\"distance_bt\":" << point.getDistance()
                      << ",\"breaking_threshold_bt\":"
                      << manifold->getContactBreakingThreshold()
                      << ",\"lifetime\":" << point.getLifeTime()
                      << ",\"applied_impulse\":" << point.m_appliedImpulse
                      << ",\"lateral_impulse_1\":"
                      << point.m_appliedImpulseLateral1
                      << ",\"lateral_impulse_2\":"
                      << point.m_appliedImpulseLateral2
                      << ",\"friction\":" << point.m_combinedFriction
                      << ",\"restitution\":" << point.m_combinedRestitution
                      << ",\"index_1\":" << point.m_index1
                      << ",\"point_a\":";
            PrintVec(point.getPositionWorldOnA());
            std::cout << ",\"point_b\":";
            PrintVec(point.getPositionWorldOnB());
            std::cout << ",\"stored_local_a\":";
            PrintVec(point.m_localPointA);
            std::cout << ",\"stored_local_b\":";
            PrintVec(point.m_localPointB);
            std::cout << ",\"normal_b\":";
            PrintVec(point.m_normalWorldOnB);
            std::cout << ",\"friction_dir_1\":";
            PrintVec(point.m_lateralFrictionDir1);
            std::cout << '}';
        }
    }
    std::cout << "]}\n";
}

void PrintDispatcherManifolds(int tick, Car* car, Arena* arena) {
    auto* dispatcher = arena->_bulletWorld.getDispatcher();
    std::cout << "{\"record\":\"dispatcher_manifolds\",\"tick\":" << tick
              << ",\"manifolds\":[";
    for (int manifoldIndex = 0; manifoldIndex < dispatcher->getNumManifolds(); ++manifoldIndex) {
        if (manifoldIndex) {
            std::cout << ',';
        }
        btPersistentManifold* manifold =
            dispatcher->getManifoldByIndexInternal(manifoldIndex);
        const auto printBody = [&](const btCollisionObject* body) {
            const char* kind = "other";
            int worldBodyIndex = -1;
            if (body == &car->_rigidBody) {
                kind = "car";
            } else if (traceOtherCar && body == &traceOtherCar->_rigidBody) {
                kind = "car_b";
            } else if (body == &arena->ball->_rigidBody) {
                kind = "ball";
            } else {
                for (int index = 0; index < arena->_worldCollisionRBs.size(); ++index) {
                    if (body == arena->_worldCollisionRBs[index]) {
                        kind = "world";
                        worldBodyIndex = index;
                        break;
                    }
                }
            }
            std::cout << "{\"kind\":\"" << kind
                      << "\",\"world_body_index\":" << worldBodyIndex
                      << ",\"island_tag\":" << body->getIslandTag()
                      << ",\"proxy_id\":"
                      << body->getBroadphaseHandle()->m_uniqueId << '}';
        };
        std::cout << "{\"dispatcher_index\":" << manifoldIndex
                  << ",\"contact_count\":" << manifold->getNumContacts()
                  << ",\"body_0\":";
        printBody(static_cast<const btCollisionObject*>(manifold->getBody0()));
        std::cout << ",\"body_1\":";
        printBody(static_cast<const btCollisionObject*>(manifold->getBody1()));
        std::cout << '}';
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

void PrintExtraHitReplay(int tick, Car* car, Ball* ball, Arena* arena) {
    using namespace RLConst;
    const CarState carState = car->GetState();
    const BallState ballState = ball->GetState();
    const Vec carForward = car->GetForwardDir();
    const Vec relativePosition = ballState.pos - carState.pos;
    const Vec relativeVelocity = ballState.vel - carState.vel;
    const float relativeSpeed = RS_MIN(
        relativeVelocity.Length(), BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL_UU);
    Vec firstDirection;
    float forwardDot = 0.0f;
    Vec forwardAdjustment;
    Vec finalDirection;
    float factor = 0.0f;
    Vec addedVelocity;
    if (relativeSpeed > 0.0f) {
        firstDirection = (
            relativePosition * Vec(1.0f, 1.0f, BALL_CAR_EXTRA_IMPULSE_Z_SCALE)
        ).Normalized();
        forwardDot = firstDirection.Dot(carForward);
        forwardAdjustment = carForward * forwardDot
            * (1.0f - BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE);
        finalDirection = (firstDirection - forwardAdjustment).Normalized();
        factor = BALL_CAR_EXTRA_IMPULSE_FACTOR_CURVE.GetOutput(relativeSpeed);
        addedVelocity = (finalDirection * relativeSpeed) * factor
            * arena->_mutatorConfig.ballHitExtraForceScale;
    }
    std::cout << "{\"record\":\"extra_hit_replay\",\"tick\":" << tick
              << ",\"car_pos\":";
    PrintVec(carState.pos);
    std::cout << ",\"car_vel\":";
    PrintVec(carState.vel);
    std::cout << ",\"ball_pos\":";
    PrintVec(ballState.pos);
    std::cout << ",\"ball_vel\":";
    PrintVec(ballState.vel);
    std::cout << ",\"car_forward\":";
    PrintVec(carForward);
    std::cout << ",\"relative_position\":";
    PrintVec(relativePosition);
    std::cout << ",\"relative_velocity\":";
    PrintVec(relativeVelocity);
    std::cout << ",\"relative_speed\":" << relativeSpeed
              << ",\"first_direction\":";
    PrintVec(firstDirection);
    std::cout << ",\"forward_dot\":" << forwardDot
              << ",\"forward_adjustment\":";
    PrintVec(forwardAdjustment);
    std::cout << ",\"final_direction\":";
    PrintVec(finalDirection);
    std::cout << ",\"factor\":" << factor
              << ",\"force_scale\":"
              << arena->_mutatorConfig.ballHitExtraForceScale
              << ",\"added_velocity\":";
    PrintVec(addedVelocity);
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

void PrintPlaneTransforms(Arena* arena) {
    for (int index = 0; index < arena->_worldCollisionRBs.size(); ++index) {
        const btCollisionShape* shape =
            arena->_worldCollisionRBs[index]->getCollisionShape();
        if (shape->getShapeType() != STATIC_PLANE_PROXYTYPE) {
            continue;
        }
        const auto* plane = static_cast<const btStaticPlaneShape*>(shape);
        const btTransform transform =
            arena->_worldCollisionRBs[index]->getWorldTransform();
        std::cout << "{\"record\":\"plane_transform\",\"world_body_index\":"
                  << index << ",\"origin\":";
        PrintVec(transform.getOrigin());
        std::cout << ",\"basis\":[";
        PrintVec(transform.getBasis()[0]);
        std::cout << ',';
        PrintVec(transform.getBasis()[1]);
        std::cout << ',';
        PrintVec(transform.getBasis()[2]);
        std::cout << "],\"normal\":";
        PrintVec(plane->getPlaneNormal());
        std::cout << ",\"constant\":" << plane->getPlaneConstant()
                  << "}\n";
    }
}

void PrintPredictedBroadphaseAabb(
    const char* kind,
    const btRigidBody& body,
    btScalar timeStep,
    int tick) {
    btVector3 minimum;
    btVector3 maximum;
    body.getCollisionShape()->getAabb(
        body.getWorldTransform(), minimum, maximum);
    const btVector3 threshold(
        gContactBreakingThreshold,
        gContactBreakingThreshold,
        gContactBreakingThreshold);
    minimum -= threshold;
    maximum += threshold;

    btVector3 dampedLinear = body.getLinearVelocity();
    btVector3 dampedAngular = body.getAngularVelocity();
    dampedLinear *= btPow(
        btScalar(1) - body.getLinearDamping(), timeStep);
    dampedAngular *= btPow(
        btScalar(1) - body.getAngularDamping(), timeStep);
    btTransform predicted;
    btTransformUtil::integrateTransform(
        body.getWorldTransform(),
        dampedLinear,
        dampedAngular,
        timeStep,
        predicted);
    btVector3 predictedMinimum;
    btVector3 predictedMaximum;
    body.getCollisionShape()->getAabb(
        predicted, predictedMinimum, predictedMaximum);
    predictedMinimum -= threshold;
    predictedMaximum += threshold;
    minimum.setMin(predictedMinimum);
    maximum.setMax(predictedMaximum);

    std::cout << "{\"record\":\"predicted_broadphase_aabb\",\"tick\":"
              << tick << ",\"kind\":\"" << kind << "\",\"minimum\":";
    PrintVec(minimum);
    std::cout << ",\"maximum\":";
    PrintVec(maximum);
    std::cout << ",\"predicted_origin\":";
    PrintVec(predicted.getOrigin());
    std::cout << ",\"predicted_basis\":[";
    PrintVec(predicted.getBasis()[0]);
    std::cout << ',';
    PrintVec(predicted.getBasis()[1]);
    std::cout << ',';
    PrintVec(predicted.getBasis()[2]);
    std::cout << "]}\n";
}

void PrintPositiveXPlaneSupport(Car* car, Arena* arena, int tick) {
    const btRigidBody* planeBody = nullptr;
    const btStaticPlaneShape* planeShape = nullptr;
    int planeBodyIndex = -1;
    for (int index = 0; index < arena->_worldCollisionRBs.size(); ++index) {
        const btCollisionShape* shape =
            arena->_worldCollisionRBs[index]->getCollisionShape();
        if (shape->getShapeType() != STATIC_PLANE_PROXYTYPE) {
            continue;
        }
        const auto* candidate = static_cast<const btStaticPlaneShape*>(shape);
        const btVector3 normal = candidate->getPlaneNormal();
        if (normal.x() == btScalar(-1) && normal.y() == btScalar(0)
            && normal.z() == btScalar(0)) {
            planeBody = arena->_worldCollisionRBs[index];
            planeShape = candidate;
            planeBodyIndex = index;
            break;
        }
    }
    if (!planeBody || !planeShape) {
        return;
    }

    // Mirror btConvexPlaneCollisionAlgorithm::processCollision without
    // mutating the dispatcher.  Keeping each btTransform intermediate intact
    // makes the native SIMD operation order observable to the GPU port.
    const btTransform bodyWorld = car->_rigidBody.getWorldTransform();
    const btTransform childLocal = car->_compoundShape.getChildTransform(0);
    const btTransform convexWorld = bodyWorld * childLocal;
    const btTransform planeWorld = planeBody->getWorldTransform();
    const btTransform planeInConvex = convexWorld.inverse() * planeWorld;
    const btTransform convexInPlane = planeWorld.inverse() * convexWorld;
    const btVector3 planeNormal = planeShape->getPlaneNormal();
    const btScalar planeConstant = planeShape->getPlaneConstant();
    const btVector3 localDirection =
        planeInConvex.getBasis() * -planeNormal;
    const btVector3 localSupport =
        car->_childHitboxShape.localGetSupportingVertex(localDirection);
    const btVector3 supportInPlane = convexInPlane(localSupport);
    const btScalar distance =
        planeNormal.dot(supportInPlane) - planeConstant;
    const btVector3 projectedInPlane =
        supportInPlane - distance * planeNormal;
    const btVector3 projectedWorld = planeWorld(projectedInPlane);

    std::cout << "{\"record\":\"positive_x_plane_support\",\"tick\":"
              << tick << ",\"world_body_index\":" << planeBodyIndex
              << ",\"body_origin\":";
    PrintVec(bodyWorld.getOrigin());
    std::cout << ",\"body_basis\":[";
    PrintVec(bodyWorld.getBasis()[0]);
    std::cout << ',';
    PrintVec(bodyWorld.getBasis()[1]);
    std::cout << ',';
    PrintVec(bodyWorld.getBasis()[2]);
    std::cout << "],\"child_local_origin\":";
    PrintVec(childLocal.getOrigin());
    std::cout << ",\"convex_world_origin\":";
    PrintVec(convexWorld.getOrigin());
    std::cout << ",\"convex_world_basis\":[";
    PrintVec(convexWorld.getBasis()[0]);
    std::cout << ',';
    PrintVec(convexWorld.getBasis()[1]);
    std::cout << ',';
    PrintVec(convexWorld.getBasis()[2]);
    std::cout << "],\"plane_world_origin\":";
    PrintVec(planeWorld.getOrigin());
    std::cout << ",\"plane_world_basis\":[";
    PrintVec(planeWorld.getBasis()[0]);
    std::cout << ',';
    PrintVec(planeWorld.getBasis()[1]);
    std::cout << ',';
    PrintVec(planeWorld.getBasis()[2]);
    std::cout << "],\"plane_normal\":";
    PrintVec(planeNormal);
    std::cout << ",\"plane_constant\":" << planeConstant
              << ",\"plane_in_convex_origin\":";
    PrintVec(planeInConvex.getOrigin());
    std::cout << ",\"plane_in_convex_basis\":[";
    PrintVec(planeInConvex.getBasis()[0]);
    std::cout << ',';
    PrintVec(planeInConvex.getBasis()[1]);
    std::cout << ',';
    PrintVec(planeInConvex.getBasis()[2]);
    std::cout << "],\"convex_in_plane_origin\":";
    PrintVec(convexInPlane.getOrigin());
    std::cout << ",\"convex_in_plane_basis\":[";
    PrintVec(convexInPlane.getBasis()[0]);
    std::cout << ',';
    PrintVec(convexInPlane.getBasis()[1]);
    std::cout << ',';
    PrintVec(convexInPlane.getBasis()[2]);
    std::cout << "],\"local_direction\":";
    PrintVec(localDirection);
    std::cout << ",\"local_support\":";
    PrintVec(localSupport);
    std::cout << ",\"support_in_plane\":";
    PrintVec(supportInPlane);
    std::cout << ",\"distance\":" << distance
              << ",\"projected_in_plane\":";
    PrintVec(projectedInPlane);
    std::cout << ",\"projected_world\":";
    PrintVec(projectedWorld);
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

void PrintCarBallGjkProbe(Car* car, Arena* arena, int tick) {
    const auto* sphere = static_cast<const btConvexShape*>(
        arena->ball->_rigidBody.getCollisionShape());
    const btTransform transformA = car->_rigidBody.getWorldTransform()
        * car->_compoundShape.getChildTransform(0);
    const btTransform transformB = arena->ball->_rigidBody.getWorldTransform();
    btVoronoiSimplexSolver simplex;
    btGjkEpaPenetrationDepthSolver penetrationDepthSolver;
    btGjkPairDetector detector(
        &car->_childHitboxShape, sphere, &simplex, &penetrationDepthSolver);
    btGjkPairDetector::ClosestPointInput input;
    const btScalar maximumDistance = car->_childHitboxShape.getMargin()
        + sphere->getMargin()
        + car->_compoundShape.getContactBreakingThreshold(gContactBreakingThreshold);
    input.m_maximumDistanceSquared = maximumDistance * maximumDistance;
    input.m_transformA = transformA;
    input.m_transformB = transformB;
    CaptureClosestPoint result;
    detector.getClosestPoints(input, result);

    btVoronoiSimplexSolver traceSimplex;
    traceSimplex.reset();
    btTransform localA = transformA;
    btTransform localB = transformB;
    const btVector3 positionOffset =
        (localA.getOrigin() + localB.getOrigin()) * btScalar(0.5);
    localA.getOrigin() -= positionOffset;
    localB.getOrigin() -= positionOffset;
    btVector3 axis(0, 1, 0);
    btScalar squaredDistance = BT_LARGE_FLOAT;
    std::cout << "{\"record\":\"car_ball_gjk_iterations\",\"tick\":" << tick
              << ",\"position_offset\":";
    PrintVec(positionOffset);
    std::cout << ",\"steps\":[";
    for (int iteration = 0; iteration < 1002; ++iteration) {
        const btVector3 directionA = (-axis) * transformA.getBasis();
        const btVector3 directionB = axis * transformB.getBasis();
        const btVector3 localP =
            car->_childHitboxShape.localGetSupportVertexWithoutMarginNonVirtual(
                directionA);
        const btVector3 localQ =
            sphere->localGetSupportVertexWithoutMarginNonVirtual(directionB);
        const btVector3 p = localA(localP);
        const btVector3 q = localB(localQ);
        const btVector3 w = p - q;
        const btScalar delta = axis.dot(w);
        const bool maximumDistanceExit = delta > btScalar(0.0)
            && delta * delta > squaredDistance * input.m_maximumDistanceSquared;
        const bool repeated = traceSimplex.inSimplex(w);
        const btScalar f0 = squaredDistance - delta;
        const btScalar f1 = squaredDistance * btScalar(1.0e-6);
        if (iteration) {
            std::cout << ',';
        }
        std::cout << "{\"iteration\":" << iteration << ",\"axis\":";
        PrintVec(axis);
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
                  << ",\"squared_distance_before\":" << squaredDistance;
        if (maximumDistanceExit || repeated || f0 <= f1) {
            std::cout << ",\"exit_reason\":\""
                      << (maximumDistanceExit ? "maximum_distance"
                          : (repeated ? "in_simplex" : "relative_progress"))
                      << "\"}";
            break;
        }
        traceSimplex.addVertex(w, p, q);
        btVector3 nextAxis;
        const bool closest = traceSimplex.closest(nextAxis);
        std::cout << ",\"closest\":" << (closest ? "true" : "false")
                  << ",\"next_axis\":";
        PrintVec(nextAxis);
        std::cout << ",\"squared_distance_after\":" << nextAxis.length2()
                  << ",\"cached_point_a\":";
        PrintVec(traceSimplex.m_cachedP1);
        std::cout << ",\"cached_point_b\":";
        PrintVec(traceSimplex.m_cachedP2);
        std::cout << ",\"simplex_count\":" << traceSimplex.numVertices()
                  << ",\"barycentric\":["
                  << traceSimplex.m_cachedBC.m_barycentricCoords[0] << ','
                  << traceSimplex.m_cachedBC.m_barycentricCoords[1] << ','
                  << traceSimplex.m_cachedBC.m_barycentricCoords[2] << ','
                  << traceSimplex.m_cachedBC.m_barycentricCoords[3] << ']';
        if (!closest || nextAxis.length2() < btScalar(1.0e-6)
            || squaredDistance - nextAxis.length2()
                <= SIMD_EPSILON * squaredDistance
            || traceSimplex.fullSimplex()) {
            std::cout << ",\"exit_reason\":\"post_closest\"}";
            break;
        }
        squaredDistance = nextAxis.length2();
        axis = nextAxis;
        std::cout << ",\"exit_reason\":null}";
    }
    std::cout << "]}\n";
    std::cout << "{\"record\":\"car_ball_gjk_probe\",\"tick\":" << tick
              << ",\"margin_a\":" << detector.m_marginA
              << ",\"margin_b\":" << detector.m_marginB
              << ",\"iterations\":" << detector.m_curIter
              << ",\"degenerate\":" << detector.m_degenerateSimplex
              << ",\"method\":" << detector.m_lastUsedMethod
              << ",\"axis\":";
    PrintVec(detector.m_cachedSeparatingAxis);
    std::cout << ",\"found\":" << (result.found ? "true" : "false")
              << ",\"normal\":";
    PrintVec(result.normal);
    std::cout << ",\"point_b\":";
    PrintVec(result.point);
    std::cout << ",\"distance\":" << result.distance << "}\n";
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
        const bool ballCustom = scenarioName == "ball_custom";
        const bool carBallCustom = scenarioName == "car_ball_custom"
            || scenarioName == "car_ball_custom_grounded";
        const bool carCarAuthority = scenarioName == "car_car_authority";
        const bool carCarCustom = scenarioName == "car_car_custom";
        const bool carCarInput = carCarCustom || carCarAuthority;
        const bool customScenario =
            scenarioName == "custom" || scenarioName == "custom_full"
            || customProbe || carBallCustom || carCarInput;
        const int ticks = std::stoi(argv[3]);
        if (ticks < 0 || ticks > 600) {
            throw std::runtime_error("ticks must be in [0, 600]");
        }

        RocketSim::Init(collisionRoot, true);
        ArenaConfig config;
        config.noBallRot = false;
        Arena* arena = Arena::Create(GameMode::SOCCAR, config, 120.0f);
        arena->SetCarCarCollision(carCarInput);
        arena->SetCarBallCollision(carBallCustom);
        Car* car = ballCustom
            ? nullptr
            : arena->AddCar(Team::BLUE, CAR_CONFIG_OCTANE);
        TracingVehicleRaycaster tracingVehicleRaycaster(&arena->_bulletWorld);
        if (car) {
            tracingVehicleRaycaster.addedFilterMask =
                car->_bulletVehicleRaycaster.addedFilterMask;
        }
        if ((customScenario && !carCarAuthority) || ballCustom) {
            if (car) {
                car->_bulletVehicle.m_vehicleRaycaster = &tracingVehicleRaycaster;
            }
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
        Car* remote = carCarInput
            ? arena->AddCar(Team::ORANGE, CAR_CONFIG_OCTANE)
            : customScenario || ballCustom
                ? nullptr
                : arena->AddCar(Team::ORANGE, CAR_CONFIG_OCTANE);
        traceCar = car;
        traceOtherCar = carCarInput ? remote : nullptr;
        traceBody = ballCustom ? &arena->ball->_rigidBody : &car->_rigidBody;
        traceBallBody = carCarInput
            ? &remote->_rigidBody
            : &arena->ball->_rigidBody;
        traceArena = arena;
        if (carCarInput) {
            std::cout << "{\"record\":\"car_iteration_order\",\"cars\":[";
            bool firstCar = true;
            for (Car* orderedCar : arena->GetCars()) {
                if (!firstCar) {
                    std::cout << ',';
                }
                firstCar = false;
                std::cout << '"' << (orderedCar == car ? "car_a" : "car_b") << '"';
            }
            std::cout << "]}\n";
        }
        priorContactAddedCallback = gContactAddedCallback;
        if (!carCarAuthority)
            gContactAddedCallback = &TraceContactAdded;

        Scenario scenario = {};
        if (!ballCustom) {
            scenario = carCarInput
                ? MakeCarCarCustomScenario(argc, argv, 0)
                : customScenario
                ? MakeCustomScenario(
                    carBallCustom ? 25 : argc, argv)
                : MakeScenario(scenarioName);
            if (scenarioName == "car_ball_custom_grounded") {
                scenario.onGround = true;
            }
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
        }

        if (carCarInput) {
            const Scenario remoteScenario =
                MakeCarCarCustomScenario(argc, argv, 1);
            CarState remoteState;
            remoteState.pos = remoteScenario.pos;
            remoteState.vel = remoteScenario.vel;
            remoteState.angVel = remoteScenario.angVel;
            remoteState.rotMat = remoteScenario.rotMat;
            remoteState.isOnGround = remoteScenario.onGround;
            remoteState.handbrakeVal = 0.0f;
            remoteState.boost = remoteScenario.boost;
            remoteState.lastControls = remoteScenario.controls;
            remote->SetState(remoteState);
            remote->controls = remoteScenario.controls;
        } else if (remote) {
            CarState remoteState;
            remoteState.pos = Vec(2500, -2500, 17);
            remoteState.isOnGround = true;
            remote->SetState(remoteState);
        }

        BallState ballState;
        ballState = ballCustom
            ? MakeCustomBallState(argc, argv)
            : (carBallCustom ? MakeCarBallCustomBallState(argc, argv) : BallState());
        if (!ballCustom && !carBallCustom) {
            ballState.pos = carCarInput
                ? Vec(-3000, -4000, 1500)
                : Vec(0, 0, 1500);
        }
        arena->ball->SetState(ballState);

        std::cout << std::setprecision(9);
        if (ballCustom) {
            PrintBallState("initial", 0, arena->ball, arena);
        } else {
            PrintShape(car);
            PrintPlaneTransforms(arena);
            PrintState("initial", 0, car, arena);
            if (carCarInput) {
                PrintState("initial_car_b", 0, remote, arena);
            }
            if (carBallCustom) {
                PrintBallState("initial_ball", 0, arena->ball, arena);
            }
        }
        const bool staged = scenarioName == "custom" || customProbe || ballCustom
            || carBallCustom || carCarCustom
            || (argc >= 5 && std::string(argv[4]) == "staged");
        for (int tick = 1; tick <= ticks; ++tick) {
            traceTick = tick;
            solverRowCall = 0;
            if (staged) {
                arena->_bulletWorld.setWorldUserInfo(arena);
                if (carBallCustom) {
                    btVector3 proxyMinimum;
                    btVector3 proxyMaximum;
                    arena->_bulletWorld.getBroadphase()->getAabb(
                        arena->ball->_rigidBody.getBroadphaseHandle(),
                        proxyMinimum,
                        proxyMaximum);
                    std::cout << "{\"record\":\"ball_broadphase_proxy\",\"tick\":"
                              << tick << ",\"minimum\":";
                    PrintVec(proxyMinimum);
                    std::cout << ",\"maximum\":";
                    PrintVec(proxyMaximum);
                    const auto* rsProxy = static_cast<const btRSBroadphaseProxy*>(
                        arena->ball->_rigidBody.getBroadphaseHandle());
                    std::cout << ",\"cell_index\":" << rsProxy->cellIdx
                              << ",\"cell_i\":" << rsProxy->iIdx
                              << ",\"cell_j\":" << rsProxy->jIdx
                              << ",\"cell_k\":" << rsProxy->kIdx;
                    std::cout << ",\"transform_origin\":";
                    PrintVec(arena->ball->_rigidBody.getWorldTransform().getOrigin());
                    std::cout << "}\n";
                }
                // Arena::Step performs this before every pre-tick hook. Keep
                // staged car/ball traces on the identical Bullet activation
                // path so a motionless ball skips applyGravity until contact
                // wakes it later in the step.
                if (arena->ball->_rigidBody.m_linearVelocity.length2() == 0
                    && arena->ball->_rigidBody.m_angularVelocity.length2() == 0) {
                    arena->ball->_rigidBody.setActivationState(ISLAND_SLEEPING);
                } else {
                    arena->ball->_rigidBody.setActivationState(ACTIVE_TAG);
                }
                if (ballCustom) {
                    arena->ball->_PreTickUpdate(
                        arena->gameMode, arena->tickTime);
                    PrintBallState("after_ball_pre", tick, arena->ball, arena);
                    arena->_bulletWorld.stepSimulation(
                        arena->tickTime, 0, arena->tickTime);
                    PrintBallState("after_bullet", tick, arena->ball, arena);
                    arena->ball->_FinishPhysicsTick(arena->_mutatorConfig);
                    arena->tickCount++;
                    PrintBallState("post", tick, arena->ball, arena);
                    continue;
                }
                const btVector3 preVehicleLinear = car->_rigidBody.getLinearVelocity();
                const btVector3 preVehicleAngular = car->_rigidBody.getAngularVelocity();
                const btVector3 preRemoteLinear = remote
                    ? remote->_rigidBody.getLinearVelocity()
                    : btVector3(0, 0, 0);
                const btVector3 preRemoteAngular = remote
                    ? remote->_rigidBody.getAngularVelocity()
                    : btVector3(0, 0, 0);
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
                        "car_a",
                        preVehicleLinear,
                        preVehicleAngular,
                        arena->tickTime,
                        tick);
                    if (remote) {
                        PrintWheelApplyReplay(
                            remote,
                            "car_b",
                            preRemoteLinear,
                            preRemoteAngular,
                            arena->tickTime,
                            tick);
                    }
                }
                PrintState("after_car_pre", tick, car, arena);
                if (carCarInput) {
                    PrintState("after_car_b_pre", tick, remote, arena);
                    PrintCarPairTransforms(tick, car, remote);
                    PrintCarPairFaceAxes(tick, car, remote);
                    PrintCarPairEdge13(tick, car, remote);
                    PrintCarPairBroadphasePrediction(
                        tick, car, remote, arena->tickTime);
                }
                if (carBallCustom && tick <= 3) {
                    PrintPredictedBroadphaseAabb(
                        "car", car->_rigidBody, arena->tickTime, tick);
                    PrintPredictedBroadphaseAabb(
                        "ball", arena->ball->_rigidBody, arena->tickTime, tick);
                }
                if (carBallCustom && tick <= 3) {
                    PrintPositiveXPlaneSupport(car, arena, tick);
                }
                if (scenarioName == "custom" || customProbe) {
                    if (!customProbe) {
                        PrintBvhTraversal(car, arena, tick);
                    }
                    for (const auto& [bodyIndex, faceIndex] : customGjkProbes) {
                        PrintGjkProbe(car, arena, bodyIndex, faceIndex, tick);
                        PrintEpaProbe(car, arena, bodyIndex, faceIndex, tick);
                    }
                }
                if (carBallCustom) {
                    PrintCarBallGjkProbe(car, arena, tick);
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
                if (carCarInput) {
                    PrintCarPairBroadphaseResident(tick, car, remote, arena);
                }
                if (carBallCustom || carCarCustom) {
                    PrintDispatcherManifolds(tick, car, arena);
                }
                PrintState("after_bullet", tick, car, arena);
                if (carCarInput) {
                    PrintState("after_bullet_car_b", tick, remote, arena);
                }
                if (carBallCustom) {
                    PrintBallState("after_bullet_ball", tick, arena->ball, arena);
                }
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
            if (carCarInput) {
                PrintState("post_car_b", tick, remote, arena);
            }
        }
        gContactAddedCallback = priorContactAddedCallback;
        traceCar = nullptr;
        traceOtherCar = nullptr;
        traceBody = nullptr;
        traceBallBody = nullptr;
        traceArena = nullptr;
        traceTick = 0;
        delete arena;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
