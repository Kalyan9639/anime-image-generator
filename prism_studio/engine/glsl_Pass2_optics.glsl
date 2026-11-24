#version 330 core

in vec2 v_uv;
out vec4 out_frag;

uniform sampler2D u_hdr_rim;
uniform sampler2D u_nrm_mat;
uniform sampler2D u_depth_hit;

uniform vec2 u_res;
uniform float u_focus_depth;
uniform float u_dof_strength;
uniform float u_ink_strength;
uniform vec2 u_sun_uv;
uniform vec3 u_sun_col;
uniform float u_godray_strength;
uniform float u_bloom_intensity;
uniform float u_chrom_ab;
uniform float u_film_grain;
uniform float u_exposure;
uniform int u_style_mode;
uniform int u_bokeh_count;
uniform int u_dust_count;
uniform int u_seed;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float safePow(float x, float y) {
    return (x <= 1e-6) ? 0.0 : pow(x, y);
}

vec3 acesTonemap(vec3 hdr, float expVal) {
    vec3 x = max(hdr * expVal, vec3(0.0));
    vec3 mapped = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
    return pow(max(mapped, vec3(1e-6)), vec3(1.0 / 2.2));
}

void main() {
    vec2 px = 1.0 / u_res;
    vec4 centerHdr = texture(u_hdr_rim, v_uv);
    vec4 centerNrm = texture(u_nrm_mat, v_uv);
    vec2 centerDH = texture(u_depth_hit, v_uv).rg;

    float zC = centerDH.r;
    float hitC = centerDH.g;

    // ── Stage 1: Light-Modulated Sobel Ink Outlines (anime_npr) ──────────────
    vec3 baseHDR = centerHdr.rgb;
    if (u_style_mode == 0 && u_ink_strength > 0.01 && hitC > 0.7) {
        float gxZ = 0.0, gyZ = 0.0;
        vec3 gxN = vec3(0.0), gyN = vec3(0.0);
        float maxObjDiff = 0.0;

        float kx[9] = float[9](-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
        float ky[9] = float[9](-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
        int idx = 0;
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                vec2 suv = v_uv + vec2(float(i), float(j)) * px * 1.15;
                float lz = log(clamp(texture(u_depth_hit, suv).r, 0.1, 40.0));
                vec4 sn = texture(u_nrm_mat, suv);
                gxZ += lz * kx[idx];
                gyZ += lz * ky[idx];
                gxN += sn.xyz * kx[idx];
                gyN += sn.xyz * ky[idx];
                maxObjDiff = max(maxObjDiff, abs(sn.a - centerNrm.a));
                idx++;
            }
        }
        float edgeZ = smoothstep(0.22, 0.55, length(vec2(gxZ, gyZ)));
        float edgeN = smoothstep(1.15, 2.20, length(gxN) + length(gyN));
        float edgeId = smoothstep(0.015, 0.045, maxObjDiff);
        float edge = clamp(max(edgeZ, max(edgeN * 0.65, edgeId * 0.85)), 0.0, 1.0);
        // Rim light cleanly thins ink outlines so golden sunset rim bleeds naturally!
        float inkWeight = u_ink_strength * clamp(1.0 - 0.92 * centerHdr.a, 0.08, 1.0);
        vec3 inkCol = vec3(0.055, 0.035, 0.075);
        baseHDR = mix(baseHDR, inkCol, edge * inkWeight);
    }

    // ── Stage 2: Foreground-Protected 24-Tap Vogel Disk Bokeh ────────────────
    float coc = 0.0;
    if (u_dof_strength > 0.02) {
        float relZ = (zC > 500.0) ? 1.0 : clamp(abs(zC - u_focus_depth) / max(zC, 0.35), 0.0, 1.0);
        coc = relZ * u_dof_strength * (min(u_res.x, u_res.y) * 0.014);
    }

    vec3 dofHDR = baseHDR;
    if (coc > 0.65) {
        vec3 acc = baseHDR;
        float wSum = 1.0;
        const float golden = 2.39996323;
        for (int i = 0; i < 24; ++i) {
            float fi = float(i);
            float r = coc * sqrt((fi + 0.5) / 24.0);
            float ang = fi * golden;
            vec2 suv = clamp(v_uv + vec2(cos(ang), sin(ang)) * r * px, vec2(0.001), vec2(0.999));
            float sz = texture(u_depth_hit, suv).r;
            float sCoc = (sz > 500.0) ? coc : clamp(abs(sz - u_focus_depth) / max(sz, 0.35), 0.0, 1.0) * u_dof_strength * (min(u_res.x, u_res.y) * 0.014);
            // Foreground protection: reject crisp foreground samples bleeding into blurred background
            float w = (sz >= zC - 0.18 || sCoc > coc * 0.65) ? 1.0 : 0.08;
            acc += texture(u_hdr_rim, suv).rgb * w;
            wSum += w;
        }
        dofHDR = acc / wSum;
    }

    // ── Stage 3: Screen-Space 16-Tap Radial Volumetric God-Rays ──────────────
    vec3 rays = vec3(0.0);
    if (u_godray_strength > 0.01) {
        vec2 delta = (u_sun_uv - v_uv) * (1.0 / 16.0);
        vec2 curUV = v_uv;
        float decay = 1.0;
        for (int k = 0; k < 16; ++k) {
            curUV += delta;
            vec2 sd = texture(u_depth_hit, clamp(curUV, 0.001, 0.999)).rg;
            float occ = (sd.g < 0.75) ? 1.0 : 0.0;
            rays += occ * decay * u_sun_col * 0.045;
            decay *= 0.91;
        }
        float sunDist = length((v_uv - u_sun_uv) * vec2(u_res.x / u_res.y, 1.0));
        float phase = exp(-sunDist * 1.45);
        dofHDR += rays * phase * u_godray_strength;
    }

    // ── Stage 4: Background Bokeh Orbs & Floating Dust Sparkles ───────────────
    if (u_bokeh_count > 0 && hitC < 0.75) {
        float aspect = u_res.x / u_res.y;
        for (int bi = 0; bi < 40; ++bi) {
            if (bi >= u_bokeh_count) break;
            float fb = float(bi) + float(u_seed) * 0.13;
            vec2 bPos = vec2(hash12(vec2(fb, 1.7)), hash12(vec2(fb, 4.3)) * 0.85);
            float bRad = mix(0.025, 0.075, hash12(vec2(fb, 9.1)));
            float d = length((v_uv - bPos) * vec2(aspect, 1.0)) / bRad;
            if (d < 1.2) {
                float disc = clamp(1.0 - safePow(d, 4.0), 0.0, 1.0);
                float ring = exp(-safePow(d - 0.82, 2.0) * 24.0) * 0.25;
                dofHDR += u_sun_col * (disc * 0.25 + ring * 0.2) * (hitC < 0.2 ? 1.0 : 0.35);
            }
        }
    }
    if (u_dust_count > 0) {
        float aspect = u_res.x / u_res.y;
        for (int di = 0; di < 32; ++di) {
            if (di >= u_dust_count) break;
            float fd = float(di) + float(u_seed) * 0.37;
            vec2 dPos = vec2(hash12(vec2(fd, 11.3)), hash12(vec2(fd, 17.9)));
            float dRad = mix(0.004, 0.011, hash12(vec2(fd, 23.1)));
            float d = length((v_uv - dPos) * vec2(aspect, 1.0)) / dRad;
            if (d < 1.5) {
                dofHDR += u_sun_col * exp(-d * d * 2.5) * 0.85;
            }
        }
    }

    // ── Stage 5: HDR Bloom (threshold > 1.0) & Subtle Chromatic Aberration ───
    if (u_bloom_intensity > 0.01) {
        vec3 bloomAcc = vec3(0.0);
        const float golden = 2.39996323;
        for (int i = 0; i < 12; ++i) {
            float fi = float(i);
            float r = (4.0 + 18.0 * (fi / 12.0));
            float ang = fi * golden;
            vec2 suv = clamp(v_uv + vec2(cos(ang), sin(ang)) * r * px, vec2(0.001), vec2(0.999));
            vec3 sCol = texture(u_hdr_rim, suv).rgb;
            bloomAcc += max(sCol - vec3(0.95), vec3(0.0));
        }
        dofHDR += (bloomAcc / 12.0) * u_bloom_intensity * 0.65;
    }

    if (u_chrom_ab > 1e-6) {
        vec2 dirC = v_uv - vec2(0.5);
        vec2 shift = dirC * dot(dirC, dirC) * u_chrom_ab * 12.0;
        float rCh = texture(u_hdr_rim, clamp(v_uv + shift, 0.001, 0.999)).r;
        float bCh = texture(u_hdr_rim, clamp(v_uv - shift, 0.001, 0.999)).b;
        dofHDR.r = mix(dofHDR.r, rCh, 0.25);
        dofHDR.b = mix(dofHDR.b, bCh, 0.25);
    }

    // Subtle vignette + 1% luma film grain + ACES Filmic Tone Mapping
    vec2 vigUV = v_uv * 2.0 - 1.0;
    float vig = clamp(1.0 - 0.16 * dot(vigUV, vigUV), 0.78, 1.0);
    dofHDR *= vig;

    vec3 ldr = acesTonemap(dofHDR, u_exposure);
    float grain = (hash12(v_uv * u_res) - 0.5) * u_film_grain;
    ldr = clamp(ldr + vec3(grain), 0.0, 1.0);
    out_frag = vec4(ldr, 1.0);
}
