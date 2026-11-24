#version 330 core

in vec2 v_uv;

layout(location = 0) out vec4 out_hdr_rim;     // RGB = linear HDR radiance, A = rim mask
layout(location = 1) out vec4 out_nrm_mat;     // RGB = geo normal, A = obj id hash
layout(location = 2) out vec2 out_depth_hit;   // R = depth_z, G = hit_flag (1=FG, 0.5=Mid/Env, 0=Sky)

uniform sampler2D u_data;
uniform vec2 u_res;
uniform vec3 u_cam_pos;
uniform vec3 u_cam_fwd;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_tan_half_fov;
uniform float u_aspect;
uniform float u_focus_depth;
uniform int u_style_mode;      // 0 = anime_npr, 1 = stylized_pbr
uniform int u_max_steps;

uniform int u_n_clusters;
uniform int u_n_splines;
uniform int u_n_sh;
uniform int u_n_solids;
uniform int u_n_garments;
uniform int u_n_decals;
uniform int u_n_midground;
uniform int u_n_emitters;

uniform int u_off_clusters;
uniform int u_off_splines;
uniform int u_off_stations;
uniform int u_off_sh;
uniform int u_off_solids;
uniform int u_off_garments;
uniform int u_off_decals;
uniform int u_off_midground;
uniform int u_off_emitters;

// Lighting & atmosphere uniforms
uniform vec3 u_key_dir;
uniform vec3 u_key_col;
uniform vec3 u_shadow_tint;
uniform vec3 u_ambient_col;
uniform vec3 u_rim_col;
uniform float u_rim_strength;
uniform float u_sss_strength;
uniform vec3 u_sss_col;

uniform vec3 u_sky_top;
uniform vec3 u_sky_horizon;
uniform vec3 u_sky_bottom;
uniform vec3 u_sky_glow;
uniform float u_ground_enabled;
uniform float u_ground_y;
uniform vec3 u_ground_col1;
uniform vec3 u_ground_col2;
uniform vec3 u_haze_col;
uniform float u_haze_strength;

vec4 fetchTex(int idx) {
    return texelFetch(u_data, ivec2(idx & 255, idx >> 8), 0);
}

float safePow(float x, float y) {
    return (x <= 1e-6) ? 0.0 : pow(x, y);
}

vec3 safeNorm(vec3 v) {
    float l = length(v);
    return (l > 1e-6) ? (v / l) : vec3(0.0, 1.0, 0.0);
}

float hash21(vec2 p) {
    float h = sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123;
    return fract(h);
}

float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
        mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

float fbm2(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 3; ++i) {
        v += a * valueNoise(p);
        p *= 2.03;
        a *= 0.5;
    }
    return v;
}

vec2 sminCubicMat(float a, float b, float k) {
    if (k <= 1e-5) {
        return vec2(min(a, b), b < a ? 1.0 : 0.0);
    }
    float h = max(k - abs(a - b), 0.0) / k;
    float d = min(a, b) - (1.0 / 6.0) * k * h * h * h;
    float m = 0.5 * h * h;
    float eta = (a < b) ? m : (1.0 - m);
    return vec2(d, eta);
}

float smaxCubic(float a, float b, float k) {
    return -sminCubicMat(-a, -b, k).x;
}

// ── Operator 1: Swept Super-Spline with Bishop Frames & Axial Endcaps ────────
vec4 evalSplineSDF(vec3 p, int spIdx, out vec3 outTan) {
    int base = u_off_splines + spIdx * 4;
    vec4 v0 = fetchTex(base + 0);
    vec4 v1 = fetchTex(base + 1);
    int stStart = int(v1.x + 0.5);
    int stCount = int(v1.y + 0.5);

    float minDist2 = 1e12;
    int bestSeg = 0;
    float bestS = 0.0;
    vec3 bestClosest = vec3(0.0);

    vec4 prevA = fetchTex(u_off_stations + stStart * 4);
    for (int i = 0; i < 15; ++i) {
        if (i >= stCount - 1) break;
        vec4 nextA = fetchTex(u_off_stations + (stStart + i + 1) * 4);
        vec3 seg = nextA.xyz - prevA.xyz;
        float len2 = max(dot(seg, seg), 1e-10);
        float s = clamp(dot(p - prevA.xyz, seg) / len2, 0.0, 1.0);
        vec3 cPt = prevA.xyz + s * seg;
        vec3 diff = p - cPt;
        float d2 = dot(diff, diff);
        if (d2 < minDist2) {
            minDist2 = d2;
            bestSeg = i;
            bestS = s;
            bestClosest = cPt;
        }
        prevA = nextA;
    }

    int idx0 = u_off_stations + (stStart + bestSeg) * 4;
    int idx1 = idx0 + 4;
    vec4 a0 = fetchTex(idx0 + 0); vec4 b0 = fetchTex(idx0 + 1);
    vec4 c0 = fetchTex(idx0 + 2); vec4 d0 = fetchTex(idx0 + 3);
    vec4 a1 = fetchTex(idx1 + 0); vec4 b1 = fetchTex(idx1 + 1);
    vec4 c1 = fetchTex(idx1 + 2); vec4 d1 = fetchTex(idx1 + 3);

    vec3 T = normalize(mix(b0.xyz, b1.xyz, bestS));
    vec3 N = mix(c0.xyz, c1.xyz, bestS);
    N = normalize(N - dot(N, T) * T);
    vec3 B = cross(T, N);
    outTan = T;

    float ra = max(0.004, mix(a0.w, a1.w, bestS));
    float rb = max(0.004, mix(b0.w, b1.w, bestS));
    float nExp = clamp(mix(c0.w, c1.w, bestS), 0.45, 4.0);
    float tStar = (float(bestSeg) + bestS) / max(1.0, float(stCount - 1));

    vec3 delta = p - bestClosest;
    float u = dot(delta, N);
    float v = dot(delta, B);

    float wAx = 0.0;
    if (bestSeg == 0 && bestS <= 1e-4) {
        vec3 T0 = fetchTex(u_off_stations + stStart * 4 + 1).xyz;
        wAx = max(-dot(delta, T0), 0.0);
    } else if (bestSeg == stCount - 2 && bestS >= 1.0 - 1e-4) {
        vec3 T1 = fetchTex(u_off_stations + (stStart + stCount - 1) * 4 + 1).xyz;
        wAx = max(dot(delta, T1), 0.0);
    }

    float ru = abs(u) / ra;
    float rv = abs(v) / rb;
    float minAB = min(ra, rb);
    float maxR = max(ru, rv);
    float dCross;
    if (maxR < 1e-6) {
        dCross = -minAB;
    } else {
        float normCross = maxR * safePow(safePow(ru / maxR, nExp) + safePow(rv / maxR, nExp), 1.0 / nExp);
        dCross = minAB * (normCross - 1.0);
    }

    vec4 v3 = fetchTex(base + 3);
    int meta = int(v3.w + 0.5);
    float capRound = float(meta % 10) / 9.0;

    float dRaw = dCross;
    if (wAx > 0.0) {
        float rw = wAx / (minAB * max(0.25, capRound));
        float m3 = max(maxR, rw);
        float norm3 = m3 * safePow(safePow(ru / m3, nExp) + safePow(rv / m3, nExp) + safePow(rw / m3, nExp), 1.0 / nExp);
        dRaw = minAB * (norm3 - 1.0);
    }

    float theta = atan(v, u);
    return vec4(dRaw * 0.72, tStar, theta, wAx);
}

// ── Operator 3: Single-Pass Body-UV Garment Shell ────────────────────────────
float evalClothShell(float dBody, float tUV, float thetaUV, float wAx, int gIdx, float jointBend) {
    int base = u_off_garments + gIdx * 4;
    vec4 g0 = fetchTex(base + 0); // fit, thick, flare, pleatCount
    vec4 g1 = fetchTex(base + 1); // pleatAmp, ruffleAmp, foldAmp, openRad
    vec4 g3 = fetchTex(base + 3); // tMin, tMax, decalStart, decalCount

    float tPast = max(tUV - 0.35, 0.0);
    float flareD = g0.z * safePow(tPast, 1.35);
    float pleatD = 0.0;
    if (g0.w > 0.5) {
        pleatD = min(0.009, g1.x) * sin(thetaUV * g0.w) * tPast;
    }
    float ruffleD = 0.0;
    if (g1.y > 0.0005) {
        float hemNear = max(1.0 - abs(tUV - g3.y) * 4.5, 0.0);
        ruffleD = min(0.009, g1.y) * sin(12.0 * thetaUV) * hemNear;
    }
    // Gentle Lipschitz-safe cloth fold modulation + open-hem taper (keep attached shoulder/hip roots full!)
    float foldD = min(0.005, g1.z) * jointBend * exp(-4.0 * abs(tUV - 0.5)) * sin(3.0 * thetaUV + 4.0 * tUV);
    float loTaper = (g3.x > 0.04) ? smoothstep(g3.x - 0.03, g3.x + 0.12, tUV) : 1.0;
    float hiTaper = smoothstep(g3.y + 0.03, g3.y - 0.12, tUV);
    float hemTaper = loTaper * hiTaper;
    float deltaEff = max(0.004, g0.x * mix(0.45, 1.0, hemTaper) + flareD + pleatD + ruffleD + foldD);
    float dShell = dBody - deltaEff;

    float dTrimT = ((tUV > 0.5 || g3.x > 0.02) ? wAx * 0.78 : 0.0) + max(g3.x - tUV, tUV - g3.y) * 0.45;
    float dTrim = dTrimT;
    if (g1.w > 0.02) {
        float dOpen = g1.w - abs(thetaUV);
        dTrim = max(dTrimT, dOpen * 0.25);
    }
    return smaxCubic(dShell, dTrim, 0.014) * 0.68;
}

// ── Operator 2: L <= 3 Spherical Harmonics Cranial Shell ─────────────────────
float evalSHShellSDF(vec3 p, int shIdx, out vec3 pLocOut) {
    int base = u_off_sh + shIdx * 12;
    vec4 s0 = fetchTex(base + 0); // center, bvhR
    vec4 s1 = fetchTex(base + 1); // radii, smoothK
    vec4 s2 = fetchTex(base + 2); // rot_inv[0], proxy.x
    vec4 s3 = fetchTex(base + 3); // rot_inv[1], proxy.y
    vec4 s4 = fetchTex(base + 4); // rot_inv[2], proxy.z

    vec3 dp = p - s0.xyz;
    vec3 pLoc = vec3(dot(s2.xyz, dp), dot(s3.xyz, dp), dot(s4.xyz, dp));
    pLocOut = pLoc;

    vec3 radii = max(s1.xyz, vec3(1e-4));
    vec3 q = pLoc / radii;
    float rReg = sqrt(dot(q, q) + 1e-8);
    vec3 dir = q / rReg;
    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x * x, yy = y * y, zz = z * z;

    vec4 c0 = fetchTex(base + 5);
    vec4 c1 = fetchTex(base + 6);
    vec4 c2 = fetchTex(base + 7);
    vec4 c3 = fetchTex(base + 8);

    float shSum = 0.0;
    shSum += c0.x * 0.282095;
    shSum += c0.y * (0.488603 * y) + c0.z * (0.488603 * z) + c0.w * (0.488603 * x);
    shSum += c1.x * (1.092548 * x * y) + c1.y * (1.092548 * y * z) + c1.z * (0.315392 * (3.0 * zz - 1.0)) + c1.w * (1.092548 * x * z);
    shSum += c2.x * (0.546274 * (xx - yy)) + c2.y * (0.590044 * y * (3.0 * xx - yy)) + c2.z * (2.890611 * x * y * z) + c2.w * (0.457046 * y * (5.0 * zz - 1.0));
    shSum += c3.x * (0.373176 * z * (5.0 * zz - 3.0)) + c3.y * (0.457046 * x * (5.0 * zz - 1.0)) + c3.z * (1.445306 * z * (xx - yy)) + c3.w * (0.590044 * x * (xx - 3.0 * yy));

    float rTarget = clamp(1.0 + shSum, 0.42, 1.95);
    float minS = min(radii.x, min(radii.y, radii.z));
    float dSH = (rReg - rTarget) * minS * 0.55;
    return dSH;
}

// ── Operator 5: Solid Primitives & Half-Bond Staggered Masonry ───────────────
float sdRoundedBox(vec3 p, vec3 b, float r) {
    float rad = min(r, min(b.x, min(b.y, b.z)) * 0.48);
    vec3 q = abs(p) - (b - rad);
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - rad;
}

float evalSolidSDF(vec3 p, int soIdx, out vec3 outBrickId) {
    int base = u_off_solids + soIdx * 9;
    vec4 v0 = fetchTex(base + 0); // center, bvhR
    vec4 v1 = fetchTex(base + 1); // half_extents, kind
    vec4 v2 = fetchTex(base + 2); // rot0, param_a
    vec4 v3 = fetchTex(base + 3); // rot1, param_b
    vec4 v4 = fetchTex(base + 4); // rot2, smooth_k
    vec4 v5 = fetchTex(base + 5); // repeat_bounds, emissive
    vec4 v6 = fetchTex(base + 6); // repeat_pitch, meta

    vec3 dp = p - v0.xyz;
    vec3 q = vec3(dot(v2.xyz, dp), dot(v3.xyz, dp), dot(v4.xyz, dp));
    int kind = int(v1.w + 0.5);
    outBrickId = vec3(0.0);

    if (kind == 1) {
        // Half-bond staggered masonry / bounded domain repetition
        vec3 sp = max(v6.xyz, vec3(1e-4));
        vec3 bd = max(v5.xyz, vec3(0.0));
        float iy = clamp(round(q.y / sp.y), -bd.y, bd.y);
        float stagX = 0.5 * sp.x * mod(abs(iy), 2.0);
        float ix = clamp(round((q.x - stagX) / sp.x), -bd.x, bd.x);
        float iz = clamp(round(q.z / sp.z), -bd.z, bd.z);
        outBrickId = vec3(ix, iy, iz);
        vec3 ql = vec3(q.x - (ix * sp.x + stagX), q.y - iy * sp.y, q.z - iz * sp.z);
        // Subtle per-brick bevel depth variation for crisp non-CG masonry feel
        float hVar = (hash21(vec2(ix, iy)) - 0.5) * 0.014;
        ql.z += hVar;
        return sdRoundedBox(ql, v1.xyz, v2.w);
    } else if (kind == 2) {
        // Torus Wheel Assembly (with radial spoke indentations!)
        float majR = max(0.01, v2.w);
        float minR = max(0.005, v3.w);
        vec2 qt = vec2(length(q.xz) - majR, q.y);
        float dTire = length(qt) - minR;
        // Inner rim disc + spokes
        float rXZ = length(q.xz);
        float dDisc = max(rXZ - majR * 0.88, abs(q.y) - minR * 0.35);
        float ang = atan(q.z, q.x);
        float spoke = 0.015 * abs(sin(ang * 5.0));
        return min(dTire, dDisc + spoke);
    } else if (kind == 3) {
        // Superquadric
        vec3 a = max(v1.xyz, vec3(1e-4));
        float e1 = clamp(v2.w, 0.15, 2.2);
        float e2 = clamp(v3.w, 0.15, 2.2);
        vec3 aq = abs(q) / a;
        float sc = max(aq.x, max(aq.y, aq.z));
        if (sc < 1e-6) return -min(a.x, min(a.y, a.z)) * 0.72;
        vec3 qn = max(aq / sc, vec3(1e-6));
        float xy = safePow(qn.x, 2.0 / e2) + safePow(qn.z, 2.0 / e2);
        float xyz = safePow(max(xy, 1e-10), e2 / e1) + safePow(qn.y, 2.0 / e1);
        float fVal = safePow(max(xyz, 1e-10), e1 * 0.5) * sc;
        return (fVal - 1.0) * min(a.x, min(a.y, a.z)) * 0.72;
    }
    return sdRoundedBox(q, v1.xyz, v2.w);
}

// ── Scene Distance Only (Fast Sphere Tracing / Normals / Shadows / AO) ───────
float evalSceneDist(vec3 p) {
    float dMin = 1e4;
    for (int ci = 0; ci < 32; ++ci) {
        if (ci >= u_n_clusters) break;
        vec4 c0 = fetchTex(u_off_clusters + ci * 2 + 0);
        float dSph = length(p - c0.xyz) - c0.w;
        if (dSph > max(0.45, dMin + 0.12)) {
            dMin = min(dMin, dSph);
            continue;
        }
        vec4 c1 = fetchTex(u_off_clusters + ci * 2 + 1);
        int spStart = int(c1.x + 0.5);
        int spCount = int(c1.y + 0.5);
        int shIdx = int(floor(c1.z + 0.1));
        int soPacked = int(c1.w + 0.5);
        int soStart = soPacked / 1024;
        int soCount = soPacked % 1024;

        float dOrg = 1e4;
        if (shIdx >= 0 && shIdx < u_n_sh) {
            vec4 s0 = fetchTex(u_off_sh + shIdx * 12);
            float dB = length(p - s0.xyz) - s0.w;
            if (dB < 0.32) {
                vec3 tmpLoc;
                dOrg = evalSHShellSDF(p, shIdx, tmpLoc);
            } else {
                dOrg = dB;
            }
        }

        float dClothMin = 1e4;
        for (int si = 0; si < 64; ++si) {
            if (si >= spCount) break;
            int idx = spStart + si;
            vec4 sp0 = fetchTex(u_off_splines + idx * 4 + 0);
            float dB = length(p - sp0.xyz) - sp0.w;
            if (dB > max(0.32, min(dOrg, dClothMin) + 0.10)) {
                dOrg = min(dOrg, dB);
                continue;
            }
            vec3 tanDummy;
            vec4 resSp = evalSplineSDF(p, idx, tanDummy);
            float dSp = resSp.x;
            vec4 sp1 = fetchTex(u_off_splines + idx * 4 + 1);
            vec4 sp2 = fetchTex(u_off_splines + idx * 4 + 2);
            int csg = int(sp1.w + 0.5);
            int gid = int(floor(sp2.w + 0.1));
            vec4 sp3 = fetchTex(u_off_splines + idx * 4 + 3);
            int role = int(sp3.w + 0.5) / 100000;

            if (role == 4 && gid >= 0) {
                float dSk = evalClothShell(dSp, resSp.y, resSp.z, resSp.w, gid, 0.25);
                dClothMin = min(dClothMin, dSk);
            } else {
                if (csg == 0) {
                    dOrg = sminCubicMat(dOrg, dSp, sp1.z).x;
                } else {
                    dOrg = min(dOrg, dSp);
                }
                if (gid >= 0 && gid < u_n_garments) {
                    float dCl = evalClothShell(dSp, resSp.y, resSp.z, resSp.w, gid, 0.4);
                    dClothMin = sminCubicMat(dClothMin, dCl, 0.052).x;
                }
            }
        }
        dMin = min(dMin, min(dOrg, dClothMin));

        for (int oi = 0; oi < 24; ++oi) {
            if (oi >= soCount) break;
            int idx = soStart + oi;
            vec4 so0 = fetchTex(u_off_solids + idx * 9);
            float dB = length(p - so0.xyz) - so0.w;
            if (dB > max(0.35, dMin + 0.12)) {
                dMin = min(dMin, dB);
                continue;
            }
            vec3 brickId;
            float dSo = evalSolidSDF(p, idx, brickId);
            vec4 so6 = fetchTex(u_off_solids + idx * 9 + 6);
            int csg = int(so6.w + 0.5) % 10;
            if (csg == 0) {
                float sk = fetchTex(u_off_solids + idx * 9 + 4).w;
                dMin = sminCubicMat(dMin, dSo, sk).x;
            } else {
                dMin = min(dMin, dSo);
            }
        }
    }
    return dMin;
}

// ── Tetrahedron 4-Tap SDF Gradient Normal ────────────────────────────────────
vec3 calcNormal(vec3 p) {
    const float h = 0.0025;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * evalSceneDist(p + k.xyy * h) +
        k.yyx * evalSceneDist(p + k.yyx * h) +
        k.yxy * evalSceneDist(p + k.yxy * h) +
        k.xxx * evalSceneDist(p + k.xxx * h)
    );
}

// ── Ripple-Free Direct Soft Penumbra Shadow ──────────────────────────────────
float calcSoftShadow(vec3 ro, vec3 rd, float kSoft) {
    float res = 1.0;
    float t = 0.045;
    for (int i = 0; i < 22; ++i) {
        if (t > 8.0) break;
        float h = evalSceneDist(ro + rd * t);
        if (h < 0.0025) return 0.20;
        res = min(res, kSoft * h / max(t, 1e-3));
        t += clamp(h, 0.03, 0.35);
    }
    res = clamp(res, 0.20, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

// ── 5-Tap SDF Ambient Occlusion ──────────────────────────────────────────────
float calcAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float w = 0.55;
    for (int i = 0; i < 5; ++i) {
        float hr = 0.035 + 0.085 * float(i);
        float d = evalSceneDist(p + n * hr);
        occ += max(0.0, hr - d) * w;
        w *= 0.62;
    }
    return clamp(1.0 - occ * 1.6, 0.25, 1.0);
}

// ── 2D Analytical UV Eye Evaluator ───────────────────────────────────────────
float sdEllipse2(vec2 p, vec2 r) {
    return (length(p / r) - 1.0) * min(r.x, r.y);
}

float sdQuadBezier2(vec2 p, vec2 a, vec2 b, vec2 c) {
    float minD2 = 1e9;
    vec2 prev = a;
    for (int i = 1; i <= 7; ++i) {
        float t = float(i) / 7.0;
        float u = 1.0 - t;
        vec2 cur = u * u * a + 2.0 * u * t * b + t * t * c;
        vec2 e = cur - prev;
        float l2 = max(1e-8, dot(e, e));
        float proj = clamp(dot(p - prev, e) / l2, 0.0, 1.0);
        vec2 d = p - (prev + proj * e);
        minD2 = min(minD2, dot(d, d));
        prev = cur;
    }
    return sqrt(minD2);
}

void eval2DEye(vec2 uv, int eyeStyle, vec3 irisCol, vec3 lashCol, out vec3 outCol, out float outAlpha) {
    outCol = vec3(0.0);
    outAlpha = 0.0;
    if (eyeStyle == 4) {
        float dArc = sdQuadBezier2(uv, vec2(-0.75, -0.05), vec2(0.0, 0.38), vec2(0.75, -0.05));
        outAlpha = clamp(1.0 - (dArc - 0.075) * 28.0, 0.0, 1.0);
        outCol = lashCol;
        return;
    }
    float ry = (eyeStyle == 0) ? 0.80 : ((eyeStyle == 1) ? 0.84 : 0.66);
    float dScl = sdEllipse2(uv, vec2(0.80, ry));
    float mScl = clamp(1.0 - dScl * 26.0, 0.0, 1.0);

    vec2 rIris = (eyeStyle == 1) ? vec2(0.54, 0.64) : vec2(0.46, 0.58);
    float dIris = sdEllipse2(uv, rIris);
    float mIris = clamp(1.0 - dIris * 32.0, 0.0, 1.0) * mScl;

    float ang = atan(uv.y, uv.x);
    float rn = clamp(length(uv / rIris), 0.0, 1.2);
    float fibers = 0.5 + 0.5 * sin(ang * 22.0 + rn * 4.0);
    float gradT = clamp(uv.y / (2.0 * rIris.y) + 0.5, 0.0, 1.0);
    vec3 irisRgb = mix(irisCol * 0.45, irisCol * 1.35, gradT) * (0.88 + 0.22 * fibers);
    float limbal = clamp(1.0 - abs(rn - 0.94) * 14.0, 0.0, 1.0);
    irisRgb *= (1.0 - 0.45 * limbal);

    vec2 rPup = (eyeStyle == 1) ? vec2(0.11, 0.48) : ((eyeStyle == 2) ? vec2(0.17, 0.28) : vec2(0.19, 0.26));
    float mPup = clamp(1.0 - sdEllipse2(uv, rPup) * 44.0, 0.0, 1.0) * mIris;

    float c1 = clamp(1.0 - sdEllipse2(uv - vec2(-0.16, 0.22), vec2(0.20, 0.16)) * 46.0, 0.0, 1.0);
    float c2 = clamp(1.0 - sdEllipse2(uv - vec2(0.15, -0.24), vec2(0.095, 0.085)) * 50.0, 0.0, 1.0) * 0.9;
    float mCatch = max(c1, c2) * mScl;

    float dUp = sdQuadBezier2(uv, vec2(-0.88, 0.12), vec2(0.0, ry * 1.18), vec2(0.88, 0.16));
    float upThick = (uv.x > 0.2) ? 0.075 : 0.058;
    float mUp = clamp(1.0 - (dUp - upThick) * 32.0, 0.0, 1.0);
    float dLo = sdQuadBezier2(uv, vec2(-0.68, -ry * 0.80), vec2(0.0, -ry * 0.98), vec2(0.70, -ry * 0.78));
    float mLo = clamp(1.0 - (dLo - 0.022) * 34.0, 0.0, 1.0);
    float mLash = max(mUp, mLo);

    vec3 sclera = vec3(0.96, 0.97, 0.99) * (0.85 + 0.15 * clamp(1.0 - uv.y, 0.0, 1.0));
    vec3 col = mix(sclera, irisRgb, mIris);
    col = mix(col, vec3(0.04, 0.03, 0.07), mPup);
    col = mix(col, vec3(1.85, 1.9, 1.95), mCatch);
    col = mix(col, lashCol, mLash);
    outCol = col;
    outAlpha = clamp(max(mScl, mLash), 0.0, 1.0);
}

// ── Procedural Albedo Lookup at Hit Point ────────────────────────────────────
vec3 evalShaderColor(int shaderId, vec3 p, vec3 n, vec3 c1, vec3 c2, vec3 brickId) {
    vec3 an = abs(n);
    vec2 uv = (an.y >= an.x && an.y >= an.z) ? p.xz : ((an.x >= an.z) ? p.zy : p.xy);
    if (shaderId == 2) { // plush_fur
        float f = 0.65 * valueNoise(uv * 16.0) + 0.35 * valueNoise(uv * 36.0);
        return clamp(c1 * (0.9 + 0.18 * f) + c2 * (0.08 * (1.0 - f)), 0.0, 1.0);
    } else if (shaderId == 3) { // tabby_stripes
        float warp = fbm2(uv * 3.5) * 0.85;
        float stripe = clamp((sin((uv.x * 12.0 + warp * 4.0) * 3.14159) - 0.15) * 2.5, 0.0, 1.0);
        return mix(c1, c1 * 0.55 + c2 * 0.25, stripe);
    } else if (shaderId == 5) { // brick_masonry
        float bid = hash21(brickId.xy + vec2(3.1, 7.7));
        float gr = fbm2(uv * 6.5);
        return clamp(c1 * (0.76 + 0.36 * bid) * (0.9 + 0.16 * (gr - 0.5)), 0.0, 1.0);
    } else if (shaderId == 6) { // stone_voronoi
        float gr = fbm2(uv * 4.2);
        return clamp(mix(c2, c1, clamp(gr * 1.4, 0.0, 1.0)), 0.0, 1.0);
    } else if (shaderId == 7) { // foliage_leaf
        float lv = fbm2(uv * 7.5);
        return clamp(c1 * (0.78 + 0.38 * lv), 0.0, 1.0);
    } else if (shaderId == 8) { // fabric_twill
        float tw = 0.5 + 0.5 * sin((uv.x + uv.y) * 55.0);
        return clamp(c1 * (0.93 + 0.10 * tw), 0.0, 1.0);
    } else if (shaderId == 10) { // plaid_tartan
        vec2 f = fract(uv * 3.0);
        vec3 col = c1;
        if (f.x > 0.18 && f.x < 0.44) col = mix(col, c2, 0.55);
        if (f.y > 0.18 && f.y < 0.44) col = mix(col, c2, 0.55);
        return col;
    } else if (shaderId == 11) { // wood_grain
        float w = fbm2(vec2(uv.x * 2.2, uv.y * 0.9)) * 2.2;
        float r = safePow(0.5 + 0.5 * sin((uv.x * 10.0 + w * 3.0) * 3.14159), 0.7);
        return mix(c2, c1, r);
    } else if (shaderId == 12) { // brushed_metal
        float s = valueNoise(vec2(uv.x * 2.0, uv.y * 45.0));
        return clamp(c1 * (0.88 + 0.22 * s), 0.0, 1.0);
    }
    return c1;
}

// ── Resolve Material, Hair Tangent, & Proxy Normal at Final Hit Point ────────
void resolveHitSurface(
    vec3 p, vec3 nGeo,
    out vec3 outAlb, out vec3 outNShade, out vec3 outHairTan,
    out float outRough, out float outMetal, out float outEmissive,
    out float outIsHair, out float outIsFace, out float outObjId
) {
    outAlb = vec3(0.8);
    outNShade = nGeo;
    outHairTan = vec3(0.0, 1.0, 0.0);
    outRough = 0.6;
    outMetal = 0.0;
    outEmissive = 0.0;
    outIsHair = 0.0;
    outIsFace = 0.0;
    outObjId = 0.1;

    float dMin = 1e4;
    for (int ci = 0; ci < 32; ++ci) {
        if (ci >= u_n_clusters) break;
        vec4 c0 = fetchTex(u_off_clusters + ci * 2 + 0);
        if (length(p - c0.xyz) - c0.w > 0.18) continue;
        vec4 c1 = fetchTex(u_off_clusters + ci * 2 + 1);
        int spStart = int(c1.x + 0.5);
        int spCount = int(c1.y + 0.5);
        int shIdx = int(floor(c1.z + 0.1));
        int soPacked = int(c1.w + 0.5);
        int soStart = soPacked / 1024;
        int soCount = soPacked % 1024;

        float dOrg = 1e4;
        vec3 orgAlb = vec3(0.8);
        vec3 orgN = nGeo;
        vec3 orgTan = vec3(0.0, 1.0, 0.0);
        float orgRough = 0.65;
        float orgHair = 0.0;
        float orgFace = 0.0;

        if (shIdx >= 0 && shIdx < u_n_sh) {
            int base = u_off_sh + shIdx * 12;
            vec4 s0 = fetchTex(base + 0);
            if (length(p - s0.xyz) - s0.w < 0.15) {
                vec3 pLoc;
                float dSH = evalSHShellSDF(p, shIdx, pLoc);
                vec4 s1 = fetchTex(base + 1);
                vec4 s2 = fetchTex(base + 2);
                vec4 s3 = fetchTex(base + 3);
                vec4 s4 = fetchTex(base + 4);
                vec4 s9 = fetchTex(base + 9);
                vec4 s10 = fetchTex(base + 10);
                vec4 s11 = fetchTex(base + 11);

                vec3 skinCol = s9.xyz;
                vec3 dir = normalize(pLoc / max(s1.xyz, vec3(1e-4)));
                float theta = atan(dir.x, dir.z);
                float phi = asin(clamp(dir.y, -1.0, 1.0));
                int meta = int(s11.w + 0.5);
                int eyeStyle = meta / 10000;
                int hasCatNose = (meta / 1000) % 10;

                // Evaluate 2D UV Eyes on front face (+Z)
                if (dir.z > 0.25) {
                    float eyeTh = (hasCatNose == 1) ? 0.39 : 0.37;
                    float eyePh = (hasCatNose == 1) ? 0.08 : 0.04;
                    float eScale = max(0.4, s10.w);
                    float uE = (abs(theta) - eyeTh) / (0.185 * eScale);
                    float vE = (phi - eyePh) / (0.155 * eScale);
                    if (abs(uE) < 1.25 && abs(vE) < 1.25) {
                        vec3 eyeC; float eyeA;
                        eval2DEye(vec2(uE, vE), eyeStyle, s10.xyz, s11.xyz, eyeC, eyeA);
                        skinCol = mix(skinCol, eyeC, eyeA);
                    }
                    // Soft pink feline nose & cute ":3" smile
                    if (hasCatNose == 1) {
                        float dNose = length(vec2(theta * 1.15, (phi + 0.15) * 1.45));
                        float mNose = clamp(1.0 - (dNose - 0.095) * 32.0, 0.0, 1.0);
                        skinCol = mix(skinCol, vec3(0.95, 0.45, 0.52), mNose);
                        float mBlush = exp(-(pow((abs(theta) - 0.54) / 0.14, 2.0) + pow((phi + 0.12) / 0.08, 2.0)));
                        skinCol = mix(skinCol, vec3(0.98, 0.52, 0.48), mBlush * 0.28);
                        float dMouth = abs(phi + 0.26 - 0.03 * cos(theta * 18.0));
                        if (abs(theta) < 0.22) {
                            float mSmile = clamp(1.0 - (dMouth - 0.012) * 55.0, 0.0, 1.0);
                            skinCol = mix(skinCol, s11.xyz, mSmile * 0.85);
                        }
                    } else {
                        // Subtle anime nose shadow line & gentle cheek blush
                        float mBlush = exp(-(pow((abs(theta) - 0.48) / 0.15, 2.0) + pow((phi + 0.12) / 0.07, 2.0)));
                        skinCol = mix(skinCol, vec3(0.96, 0.46, 0.46), mBlush * 0.22);
                        float mouthOpen = float(meta % 100) / 99.0;
                        if (mouthOpen > 0.06) {
                            // Crisp 2D UV anime open singing/smiling mouth
                            float mx = theta / 0.15;
                            float my = (phi + 0.30) / (0.055 + 0.055 * mouthOpen);
                            float dM = length(vec2(mx, my + 0.15 * mx * mx));
                            float mMask = clamp(1.0 - (dM - 1.0) * 18.0, 0.0, 1.0);
                            vec3 mCol = (my < -0.1) ? vec3(0.88, 0.36, 0.40) : vec3(0.45, 0.12, 0.16);
                            skinCol = mix(skinCol, mCol, mMask);
                        } else if (abs(theta) < 0.14 && abs(phi + 0.30 - 0.18 * theta * theta) < 0.013) {
                            skinCol = mix(skinCol, s11.xyz, 0.65);
                        }
                    }
                }

                // Proxy normal blend in world space
                vec3 proxyLoc = vec3(s2.w, s3.w, s4.w);
                vec3 proxyW = s0.xyz + s2.xyz * proxyLoc.x + s3.xyz * proxyLoc.y + s4.xyz * proxyLoc.z;
                vec3 nProxy = normalize(p - proxyW);
                float faceMask = smoothstep(0.15, 0.55, dir.z);
                vec3 nSH = normalize(mix(nGeo, nProxy, s9.w * faceMask));

                dOrg = dSH;
                orgAlb = skinCol;
                orgN = nSH;
                orgFace = 1.0;
            }
        }

        float dClothMin = 1e4;
        vec3 clothAlb = vec3(0.8);
        float clothRough = 0.78;
        float clothSkirtMin = 1e4;
        vec3 skirtAlb = vec3(0.8);
        float skirtRough = 0.78;

        for (int si = 0; si < 64; ++si) {
            if (si >= spCount) break;
            int idx = spStart + si;
            vec4 sp0 = fetchTex(u_off_splines + idx * 4 + 0);
            if (length(p - sp0.xyz) - sp0.w > 0.24) continue;
            vec3 spTan;
            vec4 resSp = evalSplineSDF(p, idx, spTan);
            float dSp = resSp.x;
            vec4 sp1 = fetchTex(u_off_splines + idx * 4 + 1);
            vec4 sp2 = fetchTex(u_off_splines + idx * 4 + 2);
            vec4 sp3 = fetchTex(u_off_splines + idx * 4 + 3);
            int csg = int(sp1.w + 0.5);
            int gid = int(floor(sp2.w + 0.1));
            int meta = int(sp3.w + 0.5);
            int role = meta / 100000;
            int shId = (meta / 1000) % 100;
            float rgh = float((meta / 10) % 100) / 99.0;

            if (role != 4) {
                vec3 spAlb = evalShaderColor(shId, p, nGeo, sp2.xyz, sp3.xyz, vec3(0.0));
                if (csg == 0) {
                    vec2 bl = sminCubicMat(dOrg, dSp, sp1.z);
                    dOrg = bl.x;
                    orgAlb = mix(orgAlb, spAlb, bl.y);
                    orgN = normalize(mix(orgN, nGeo, bl.y));
                    orgTan = normalize(mix(orgTan, spTan, bl.y));
                    orgRough = mix(orgRough, rgh, bl.y);
                    orgHair = mix(orgHair, (role == 1) ? 1.0 : 0.0, bl.y);
                    orgFace = mix(orgFace, 0.0, bl.y);
                } else if (dSp < dOrg) {
                    dOrg = dSp;
                    orgAlb = spAlb;
                    orgN = nGeo;
                    orgTan = spTan;
                    orgRough = rgh;
                    orgHair = (role == 1) ? 1.0 : 0.0;
                    orgFace = 0.0;
                }
            }

            if (gid >= 0 && gid < u_n_garments) {
                float dCl = evalClothShell(dSp, resSp.y, resSp.z, resSp.w, gid, (role == 4) ? 0.25 : 0.4);
                int gb = u_off_garments + gid * 4;
                vec4 g2 = fetchTex(gb + 2);
                vec4 g3 = fetchTex(gb + 3);
                int gMeta = int(g2.w + 0.5);
                int gSh = gMeta / 1000;
                float gRgh = float(gMeta % 1000) / 99.0;
                vec3 clAlb = evalShaderColor(gSh, p, nGeo, g2.xyz, g2.xyz * 0.65, vec3(0.0));
                // 2D Garment Decals
                int decStart = int(g3.z + 0.5);
                int decCount = int(g3.w + 0.5);
                for (int di = 0; di < 6; ++di) {
                    if (di >= decCount) break;
                    int db = u_off_decals + (decStart + di) * 3;
                    vec4 dc0 = fetchTex(db + 0); // kind, uv_t, uv_theta, size
                    vec4 dc1 = fetchTex(db + 1); // color
                    float dt = resSp.y - dc0.y;
                    float dth = mod(resSp.z - dc0.z + 3.14159, 6.28318) - 3.14159;
                    float rDec = length(vec2(dth * 0.45, dt)) / max(0.02, dc0.w);
                    if (rDec < 1.3) {
                        float mDec = 0.0;
                        int dk = int(dc0.x + 0.5);
                        if (dk == 1) { // medical_cross
                            vec2 qd = abs(vec2(dth * 0.45, dt)) / dc0.w;
                            float dc = min(max(qd.x - 0.82, qd.y - 0.28), max(qd.x - 0.28, qd.y - 0.82));
                            mDec = clamp(1.0 - dc * 22.0, 0.0, 1.0);
                        } else if (dk == 2) { // safety_stripes
                            mDec = clamp(1.0 - (abs(dt) / dc0.w - 0.25) * 28.0, 0.0, 1.0);
                        } else {
                            mDec = clamp(1.0 - (rDec - 0.72) * 20.0, 0.0, 1.0);
                        }
                        clAlb = mix(clAlb, dc1.xyz, mDec);
                    }
                }
                if (role == 4) {
                    if (dCl < clothSkirtMin) {
                        clothSkirtMin = dCl;
                        skirtAlb = clAlb;
                        skirtRough = gRgh;
                    }
                } else {
                    vec2 cbl = sminCubicMat(dClothMin, dCl, 0.052);
                    dClothMin = cbl.x;
                    clothAlb = mix(clothAlb, clAlb, cbl.y);
                    clothRough = mix(clothRough, gRgh, cbl.y);
                }
            }
        }

        if (clothSkirtMin < dMin) {
            dMin = clothSkirtMin;
            outAlb = skirtAlb;
            outNShade = nGeo;
            outRough = skirtRough;
            outMetal = 0.0;
            outEmissive = 0.0;
            outIsHair = 0.0;
            outIsFace = 0.0;
            outObjId = float(ci + 1) * 0.08 + 0.04;
        }
        if (dClothMin < dMin) {
            dMin = dClothMin;
            outAlb = clothAlb;
            outNShade = nGeo;
            outRough = clothRough;
            outMetal = 0.0;
            outEmissive = 0.0;
            outIsHair = 0.0;
            outIsFace = 0.0;
            outObjId = float(ci + 1) * 0.08 + 0.03;
        }

        if (dOrg < dMin) {
            dMin = dOrg;
            outAlb = orgAlb;
            outNShade = orgN;
            outHairTan = orgTan;
            outRough = orgRough;
            outMetal = 0.0;
            outEmissive = 0.0;
            outIsHair = orgHair;
            outIsFace = orgFace;
            outObjId = float(ci + 1) * 0.08;
        }

        for (int oi = 0; oi < 24; ++oi) {
            if (oi >= soCount) break;
            int idx = soStart + oi;
            vec4 so0 = fetchTex(u_off_solids + idx * 9);
            if (length(p - so0.xyz) - so0.w > 0.24) continue;
            vec3 brickId;
            float dSo = evalSolidSDF(p, idx, brickId);
            if (dSo < dMin) {
                dMin = dSo;
                vec4 so5 = fetchTex(u_off_solids + idx * 9 + 5);
                vec4 so6 = fetchTex(u_off_solids + idx * 9 + 6);
                vec4 so7 = fetchTex(u_off_solids + idx * 9 + 7);
                vec4 so8 = fetchTex(u_off_solids + idx * 9 + 8);
                int meta = int(so6.w + 0.5);
                int shId = meta / 1000;
                float rgh = float((meta / 10) % 100) / 99.0;
                outAlb = evalShaderColor(shId, p, nGeo, so7.xyz, so8.xyz, brickId);
                outNShade = nGeo;
                outRough = rgh;
                outMetal = so7.w;
                outEmissive = so5.w;
                outIsHair = 0.0;
                outIsFace = 0.0;
                outObjId = float(ci + 1) * 0.08 + 0.05;
            }
        }
    }
}

// ── Zone 3 Sky Dome + HDR Emitters ───────────────────────────────────────────
vec3 evalSky(vec3 rd) {
    float t = clamp(rd.y * 0.85 + 0.25, 0.0, 1.0);
    vec3 sky = mix(u_sky_bottom, u_sky_horizon, smoothstep(0.0, 0.35, t));
    sky = mix(sky, u_sky_top, smoothstep(0.32, 0.95, t));
    for (int i = 0; i < 4; ++i) {
        if (i >= u_n_emitters) break;
        vec4 e0 = fetchTex(u_off_emitters + i * 2 + 0);
        vec4 e1 = fetchTex(u_off_emitters + i * 2 + 1);
        vec3 lDir = normalize(e0.xyz);
        float cosA = max(dot(rd, lDir), 0.0);
        float ang = acos(clamp(cosA, -1.0, 1.0));
        float disc = smoothstep(e0.w * 1.25, e0.w * 0.65, ang);
        float halo = safePow(cosA, 8.0) * 0.28 + safePow(cosA, 48.0) * 0.55;
        sky += e1.rgb * (disc * 0.85 + halo * 0.14);
    }
    return sky;
}

// ── Zone 2 Midground Ray-Box + Pitched Roof Intersection ─────────────────────
bool intersectMidground(vec3 ro, vec3 rd, out float outT, out vec3 outCol) {
    outT = 1e9;
    outCol = vec3(0.0);
    bool hit = false;
    vec3 invRd = 1.0 / mix(rd, vec3(1e-5), lessThan(abs(rd), vec3(1e-5)));

    for (int i = 0; i < 28; ++i) {
        if (i >= u_n_midground) break;
        int base = u_off_midground + i * 3;
        vec4 m0 = fetchTex(base + 0); // center, roof_pitch
        vec4 m1 = fetchTex(base + 1); // half_extents, has_win
        vec4 m2 = fetchTex(base + 2); // color_rgb, packed_win_col

        vec3 bmin = m0.xyz - m1.xyz;
        vec3 bmax = m0.xyz + m1.xyz + vec3(0.0, max(m0.w, 0.0), 0.0);
        vec3 t0 = (bmin - ro) * invRd;
        vec3 t1 = (bmax - ro) * invRd;
        vec3 tsm = min(t0, t1);
        vec3 tbg = max(t0, t1);
        float tEnter = max(tsm.x, max(tsm.y, tsm.z));
        float tExit = min(tbg.x, min(tbg.y, tbg.z));
        if (tExit >= max(tEnter, 0.2) && tEnter < outT) {
            vec3 hp = ro + rd * max(tEnter, 0.2);
            float localY = hp.y - (m0.y + m1.y);
            bool valid = true;
            if (m0.w < -1.5) {
                // Organic multi-lobed forest tree canopy silhouette + tapered curved trunk
                float relY = (hp.y - m0.y) / max(0.1, m1.y);
                float dx = (hp.x - m0.x) / max(0.1, m1.x);
                if (relY < -0.15) {
                    float trunkW = 0.15 * (1.0 - max(relY + 0.65, 0.0) * 0.35);
                    if (abs(dx - 0.05 * sin(relY * 3.0)) > trunkW) valid = false;
                } else {
                    float l1 = dx * dx * 1.15 + pow((relY - 0.45) / 0.55, 2.0);
                    float l2 = pow((dx + 0.44) / 0.58, 2.0) + pow((relY - 0.16) / 0.46, 2.0);
                    float l3 = pow((dx - 0.44) / 0.58, 2.0) + pow((relY - 0.22) / 0.46, 2.0);
                    if (min(l1, min(l2, l3)) > 1.0) valid = false;
                }
            } else if (m0.w < -0.5) {
                // Rounded standing silhouette figure (head sphere + tapered shoulders)
                float relY = (hp.y - m0.y) / max(0.1, m1.y);
                float dx = abs(hp.x - m0.x) / max(0.1, m1.x);
                if (relY > 0.55) {
                    float hd = (relY - 0.78) / 0.22;
                    if (dx * dx * 2.2 + hd * hd > 1.0) valid = false;
                } else if (dx > 0.85 - max(0.0, relY) * 0.2) {
                    valid = false;
                }
            } else if (localY > 0.0 && m0.w > 0.05) {
                float roofFrac = 1.0 - clamp(localY / m0.w, 0.0, 1.0);
                if (abs(hp.x - m0.x) > m1.x * roofFrac + 0.08) valid = false;
            }
            if (valid) {
                outT = max(tEnter, 0.2);
                hit = true;
                vec3 col = m2.xyz;
                if (m1.w > 0.5 && localY < 0.0 && m0.w >= 0.0) {
                    vec2 wf = fract(hp.xy * vec2(3.5, 2.8));
                    if (wf.x > 0.35 && wf.x < 0.65 && wf.y > 0.35 && wf.y < 0.68 && hash21(floor(hp.xy * 3.5)) > 0.45) {
                        int wc = int(m2.w + 0.5);
                        vec3 wCol = vec3(float((wc >> 16) & 255), float((wc >> 8) & 255), float(wc & 255)) / 255.0;
                        col = wCol * 1.45;
                    }
                }
                float hazeF = clamp((outT - 4.0) / 15.0, 0.0, 0.85) * u_haze_strength;
                outCol = mix(col, u_haze_col, hazeF);
            }
        }
    }
    return hit;
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(u_cam_fwd + (ndc.x * u_aspect * u_tan_half_fov) * u_cam_right + (ndc.y * u_tan_half_fov) * u_cam_up);
    vec3 ro = u_cam_pos;

    // 1. Background sky + Zone 2 midground + Ground plane
    vec3 bgCol = evalSky(rd);
    float depthZ = 1e4;
    float hitFlag = 0.0;
    vec3 finalCol = bgCol;
    vec3 finalNrm = vec3(0.0, 1.0, 0.0);
    float rimOut = 0.0;
    float objIdOut = 0.0;

    float midT; vec3 midCol;
    if (intersectMidground(ro, rd, midT, midCol)) {
        depthZ = midT;
        finalCol = midCol;
        hitFlag = 0.5;
        objIdOut = 0.5;
    }

    if (u_ground_enabled > 0.5 && rd.y < -1e-4) {
        float tG = (u_ground_y - ro.y) / rd.y;
        if (tG > 0.1 && tG < 38.0 && tG < depthZ) {
            vec3 pg = ro + rd * tG;
            float gNoise = fbm2(pg.xz * 2.2);
            vec3 gAlb = mix(u_ground_col2, u_ground_col1, gNoise);
            float shG = calcSoftShadow(pg + vec3(0.0, 0.03, 0.0), u_key_dir, 9.0);
            vec3 gLit = gAlb * (u_ambient_col * 0.45 + u_key_col * max(u_key_dir.y, 0.15) * shG * 1.2);
            float fade = smoothstep(38.0, 10.0, tG);
            finalCol = mix(bgCol, mix(gLit, u_haze_col, clamp((tG - 6.0) / 24.0, 0.0, 0.7) * u_haze_strength), fade);
            depthZ = tG;
            hitFlag = 0.5;
            finalNrm = vec3(0.0, 1.0, 0.0);
            objIdOut = 0.25;
        }
    }

    // 2. Sphere Trace Zone 1 Foreground Heroes & Architecture
    float tCur = 0.05;
    float tMax = min(depthZ + 0.1, 28.0);
    bool hitFG = false;
    for (int step = 0; step < 128; ++step) {
        if (step >= u_max_steps + 44 || tCur > tMax) break;
        vec3 p = ro + rd * tCur;
        float d = evalSceneDist(p);
        if (d < 0.0014) {
            hitFG = true;
            break;
        }
        tCur += max(d * 0.88, 0.0018);
    }

    if (hitFG && tCur < depthZ) {
        vec3 p = ro + rd * tCur;
        vec3 nGeo = calcNormal(p);
        vec3 alb, nShade, hairTan;
        float rgh, metal, emi, isHair, isFace, objId;
        resolveHitSurface(p, nGeo, alb, nShade, hairTan, rgh, metal, emi, isHair, isFace, objId);

        float sh = calcSoftShadow(p + nGeo * 0.045, u_key_dir, 8.5);
        float ao = calcAO(p, nGeo);
        vec3 V = -rd;
        float ndotl = dot(nShade, u_key_dir);
        float ndotv = clamp(dot(nGeo, V), 0.0, 1.0);
        float fresnel = safePow(1.0 - ndotv, 3.4) * u_rim_strength * (0.4 + 0.6 * ao) * mix(1.0, 0.45, isHair);

        // Dual Aesthetic Modes: 0 = anime_npr, 1 = stylized_pbr
        vec3 litRGB = vec3(0.0);
        if (u_style_mode == 0) {
            float litBand = ndotl * sh;
            float cel = smoothstep(0.14, 0.24, litBand);
            vec3 shadowCol = alb * u_shadow_tint * 0.94 * ao;
            vec3 brightCol = alb * u_key_col * 1.05;
            litRGB = mix(shadowCol, brightCol, cel);
            // Warm coral SSS terminator glow
            float sssBand = exp(-safePow((litBand - 0.19) / 0.085, 2.0)) * u_sss_strength * (0.42 + 0.58 * isFace);
            litRGB += alb * u_sss_col * sssBand;

            // Controlled Dual-lobe Kajiya-Kay Anisotropic Hair Sheen ("Angel Ring")
            if (isHair > 0.2) {
                vec3 tShift = safeNorm(hairTan + nGeo * 0.18);
                vec3 H = safeNorm(u_key_dir + V);
                float tdoth = dot(tShift, H);
                float sinTH = sqrt(max(0.0, 1.0 - tdoth * tdoth));
                float lobe1 = safePow(sinTH, 26.0) * 0.10;
                float lobe2 = safePow(sinTH, 96.0) * 0.22;
                litRGB += (alb * lobe1 + u_key_col * lobe2) * sh * isHair;
            } else {
                vec3 H = safeNorm(u_key_dir + V);
                float specMask = clamp((0.52 - rgh) / 0.35, 0.0, 1.0);
                float spec = smoothstep(0.91, 0.97, max(dot(nGeo, H), 0.0)) * specMask * sh;
                litRGB += u_key_col * spec * 0.38;
            }
        } else {
            // Stylized PBR with plush fur rim and GGX highlight
            float diff = max(ndotl, 0.0) * sh;
            float wrap = clamp((ndotl + 0.45) / 1.45, 0.0, 1.0) * u_sss_strength;
            vec3 H = safeNorm(u_key_dir + V);
            float ndoth = max(dot(nGeo, H), 0.0);
            float alphaR = max(0.06, rgh * rgh);
            float denom = ndoth * ndoth * (alphaR * alphaR - 1.0) + 1.0;
            float D_ggx = (alphaR * alphaR) / (3.14159 * denom * denom);
            float spec = D_ggx * sh * mix(0.08, 0.85, metal);
            vec3 sheen = alb * safePow(1.0 - ndotv, 4.0) * 0.45 * (1.0 - metal);
            litRGB = alb * (u_ambient_col * 0.55 * ao + u_key_col * diff * 1.35 + u_sss_col * wrap * 0.35) + u_key_col * spec + sheen;
        }

        // Cool bounce fill + HDR Rim + Emissive
        vec3 fillDir = normalize(vec3(-u_key_dir.x, 0.35, -u_key_dir.z));
        litRGB += alb * u_ambient_col * max(dot(nGeo, fillDir), 0.0) * 0.28 * ao;
        litRGB += u_rim_col * fresnel * (0.45 + 0.55 * clamp(-dot(nGeo, u_key_dir) * 0.5 + 0.5, 0.0, 1.0));
        litRGB += alb * emi * 1.9;

        depthZ = tCur;
        finalCol = litRGB;
        finalNrm = nGeo;
        hitFlag = 1.0;
        rimOut = clamp(fresnel, 0.0, 1.5);
        objIdOut = objId;
    }

    out_hdr_rim = vec4(max(finalCol, vec3(0.0)), rimOut);
    out_nrm_mat = vec4(finalNrm, objIdOut);
    out_depth_hit = vec2(depthZ, hitFlag);
}
