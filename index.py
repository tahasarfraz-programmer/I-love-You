"""
Nebula Heart 3D - Hand Gesture Particle Controller
--------------------------------------------------
Two windows, exactly like the video:

  1. "Hand Sensor Monitor"                  (OpenCV)  webcam + hand skeleton + green MODE label
  2. "Space Gesture Controller - Mac Optimized" (pygame) 700x600 particle canvas

Gestures
  Open hand (5 fingers)   -> MODE: BEBAS ANGKASA (Terbuka)      blue free-floating space dust
  Fist                    -> MODE: BENTUK HATI / LOVE (Kepal)   pink 3D rotating heart
  Peace (2 fingers)       -> MODE: TEKS: I LOVE YOU (Peace)     cyan "I LOVE YOU" text
  One finger              -> MODE: SATURNUS 3D (Satu Jari)      orange planet + particle ring

The particle shape follows your hand position. Press ESC / Q to quit.

Run:
    python index.py            # webcam mode
    python index.py --demo     # no camera: cycles through all modes automatically
"""

import math
import os
import random
import sys
import time

import numpy as np
import pygame

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
WIN_W, WIN_H = 700, 600
FPS = 60
N_PARTICLES = 1400
PARTICLE_SIZE = 4
FREE_PARTICLE_SIZE = 5
CAM_INDEX = 0
FLIP_CAMERA = True          # mirror image (selfie view)
SMOOTH_POS = 0.12           # how fast the shape follows the hand
SMOOTH_MORPH = 0.09         # how fast particles fly to their targets
SMOOTH_COLOR = 0.08

BLUE = np.array([40, 110, 230], dtype=np.float32)
PINK = np.array([255, 25, 100], dtype=np.float32)
CYAN = np.array([0, 190, 240], dtype=np.float32)
ORANGE = np.array([255, 140, 10], dtype=np.float32)
TAN = np.array([215, 160, 90], dtype=np.float32)

MODE_FREE, MODE_HEART, MODE_TEXT, MODE_SATURN = "free", "heart", "text", "saturn"

MODE_LABEL = {
    MODE_FREE: "MODE: BEBAS ANGKASA (Terbuka)",
    MODE_HEART: "MODE: BENTUK HATI / LOVE (Kepal)",
    MODE_TEXT: "MODE: TEKS: I LOVE YOU (Peace)",
    MODE_SATURN: "MODE: SATURNUS 3D (Satu Jari)",
}


# ----------------------------------------------------------------------------
# Particle engine
# ----------------------------------------------------------------------------
class ParticleSystem:
    def __init__(self, n=N_PARTICLES):
        self.n = n
        rng = np.random.default_rng(7)
        self.rng = rng
        # positions are stored relative to the shape centre (3D) except in free mode
        self.pos = np.zeros((n, 3), dtype=np.float32)
        self.pos[:, 0] = np.clip(rng.normal(0, 150, n), -WIN_W / 2, WIN_W / 2)
        self.pos[:, 1] = np.clip(rng.normal(0, 170, n), -WIN_H / 2, WIN_H / 2)
        self.pos[:, 2] = rng.uniform(-60, 60, n)
        self.target = self.pos.copy()
        self.color = np.tile(BLUE, (n, 1)).astype(np.float32)
        self.color_target = self.color.copy()
        self.vel = rng.normal(0, 0.35, (n, 3)).astype(np.float32)

        self.center = np.array([WIN_W / 2, WIN_H / 2], dtype=np.float32)
        self.center_target = self.center.copy()

        self.mode = MODE_FREE
        self.t = 0.0
        self.text_points = self._build_text_points("I LOVE YOU")
        self.heart_base = self._build_heart_points()
        self.saturn_base, self.saturn_is_ring = self._build_saturn_points()
        self.set_mode(MODE_FREE, force=True)

    # ---- shape builders ---------------------------------------------------
    def _build_heart_points(self):
        n = self.n
        t = self.rng.uniform(0, 2 * math.pi, n)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        scale = 8.3
        # outline (hollow heart) with slight thickness / jitter in every axis
        jitter = self.rng.normal(0, 7.0, (n, 3))
        pts = np.stack([x * scale, -y * scale, np.zeros(n)], axis=1)
        pts += jitter
        pts[:, 2] = self.rng.uniform(-28, 28, n)
        return pts.astype(np.float32)

    def _build_text_points(self, text):
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont("arialblack,arial,verdana", 96, bold=True)
        surf = font.render(text, True, (255, 255, 255), (0, 0, 0))
        # fit into window width
        max_w = WIN_W - 40
        if surf.get_width() > max_w:
            k = max_w / surf.get_width()
            surf = pygame.transform.smoothscale(
                surf, (int(surf.get_width() * k), int(surf.get_height() * k))
            )
        arr = pygame.surfarray.array3d(surf).sum(axis=2)  # (w, h)
        xs, ys = np.nonzero(arr > 380)
        if len(xs) == 0:
            xs = np.array([0]); ys = np.array([0])
        idx = self.rng.integers(0, len(xs), self.n)
        px = xs[idx] - surf.get_width() / 2
        py = ys[idx] - surf.get_height() / 2
        pts = np.stack([px, py, self.rng.uniform(-6, 6, self.n)], axis=1)
        pts[:, :2] += self.rng.normal(0, 0.8, (self.n, 2))
        return pts.astype(np.float32)

    def _build_saturn_points(self):
        n = self.n
        n_ball = int(n * 0.45)
        # planet: filled sphere
        u = self.rng.uniform(-1, 1, n_ball)
        th = self.rng.uniform(0, 2 * math.pi, n_ball)
        r = 56 * np.cbrt(self.rng.uniform(0.05, 1, n_ball))
        s = np.sqrt(1 - u * u)
        ball = np.stack([r * s * np.cos(th), r * s * np.sin(th), r * u], axis=1)
        # ring: wide flattened disc of dust
        m = n - n_ball
        ang = self.rng.uniform(0, 2 * math.pi, m)
        rad = 85 + 140 * np.power(self.rng.uniform(0, 1, m), 0.9)
        ring = np.stack(
            [rad * np.cos(ang), rng_n(self.rng, m, 6), rad * np.sin(ang)], axis=1
        )
        pts = np.concatenate([ball, ring]).astype(np.float32)
        is_ring = np.concatenate([np.zeros(n_ball, bool), np.ones(m, bool)])
        return pts, is_ring

    # ---- mode handling ------------------------------------------------------
    def set_mode(self, mode, force=False):
        if mode == self.mode and not force:
            return
        self.mode = mode
        if mode == MODE_FREE:
            self.color_target[:] = BLUE
        elif mode == MODE_HEART:
            self.color_target[:] = PINK
        elif mode == MODE_TEXT:
            self.color_target[:] = CYAN
        elif mode == MODE_SATURN:
            self.color_target[self.saturn_is_ring] = TAN
            self.color_target[~self.saturn_is_ring] = ORANGE
        # slightly scramble so the morph looks like a burst
        self.pos += self.rng.normal(0, 6, self.pos.shape).astype(np.float32)

    def set_center(self, x, y):
        self.center_target[:] = (x, y)

    # ---- update --------------------------------------------------------------
    def update(self, dt):
        self.t += dt
        t = self.t

        if self.mode == MODE_FREE:
            # drifting space dust spread over the whole canvas (absolute coords)
            self.pos += self.vel
            self.vel += self.rng.normal(0, 0.03, self.vel.shape).astype(np.float32)
            self.vel *= 0.995
            self.pos[:, 0] = np.where(self.pos[:, 0] < -WIN_W / 2, WIN_W / 2, self.pos[:, 0])
            self.pos[:, 0] = np.where(self.pos[:, 0] > WIN_W / 2, -WIN_W / 2, self.pos[:, 0])
            self.pos[:, 1] = np.where(self.pos[:, 1] < -WIN_H / 2, WIN_H / 2, self.pos[:, 1])
            self.pos[:, 1] = np.where(self.pos[:, 1] > WIN_H / 2, -WIN_H / 2, self.pos[:, 1])
            # in free mode the cloud only drifts a little with the hand
            self.center += (self.center_target - self.center) * SMOOTH_POS
            self.target = self.pos
        else:
            if self.mode == MODE_HEART:
                a = t * 1.25
                base = self.heart_base
                ca, sa = math.cos(a), math.sin(a)
                tgt = np.empty_like(base)
                tgt[:, 0] = base[:, 0] * ca + base[:, 2] * sa
                tgt[:, 1] = base[:, 1]
                tgt[:, 2] = -base[:, 0] * sa + base[:, 2] * ca
                beat = 1 + 0.035 * math.sin(t * 5.0)
                tgt[:, :2] *= beat
            elif self.mode == MODE_TEXT:
                tgt = self.text_points.copy()
                tgt[:, 1] += np.sin(t * 2 + tgt[:, 0] * 0.02) * 1.5
            else:  # saturn
                a = t * 0.7
                base = self.saturn_base.copy()
                ring = self.saturn_is_ring
                ca, sa = math.cos(a), math.sin(a)
                rx = base[ring, 0] * ca - base[ring, 2] * sa
                rz = base[ring, 0] * sa + base[ring, 2] * ca
                base[ring, 0] = rx
                base[ring, 2] = rz
                # tilt the whole system
                tilt = 0.38
                ct, st = math.cos(tilt), math.sin(tilt)
                y = base[:, 1] * ct - base[:, 2] * st
                z = base[:, 1] * st + base[:, 2] * ct
                base[:, 1] = y
                base[:, 2] = z
                tgt = base
            self.target = tgt
            self.pos += (self.target - self.pos) * SMOOTH_MORPH
            self.center += (self.center_target - self.center) * SMOOTH_POS

        self.color += (self.color_target - self.color) * SMOOTH_COLOR

    # ---- draw ------------------------------------------------------------------
    def draw(self, surface):
        surface.fill((0, 0, 0))
        fov = 520.0
        z = self.pos[:, 2]
        persp = fov / (fov + z)
        if self.mode == MODE_FREE:
            sx = self.pos[:, 0] + WIN_W / 2 + (self.center[0] - WIN_W / 2) * 0.25
            sy = self.pos[:, 1] + WIN_H / 2 + (self.center[1] - WIN_H / 2) * 0.25
        else:
            sx = self.pos[:, 0] * persp + self.center[0]
            sy = self.pos[:, 1] * persp + self.center[1]
        order = np.argsort(-z)
        col = np.clip(self.color, 0, 255).astype(np.int32)
        shade = np.clip(0.75 + (-z / 260.0), 0.55, 1.15) if self.mode != MODE_TEXT else np.ones_like(z)
        size = FREE_PARTICLE_SIZE if self.mode == MODE_FREE else PARTICLE_SIZE
        if self.mode == MODE_FREE:
            shade = np.ones_like(z)
        for i in order:
            c = col[i]
            s = shade[i]
            surface.fill(
                (min(255, int(c[0] * s)), min(255, int(c[1] * s)), min(255, int(c[2] * s))),
                (int(sx[i]), int(sy[i]), size, size),
            )


def rng_n(rng, m, s):
    return rng.normal(0, s, m)


# ----------------------------------------------------------------------------
# Hand tracking
# ----------------------------------------------------------------------------
class HandTracker:
    TIPS = [8, 12, 16, 20]
    PIPS = [6, 10, 14, 18]

    def __init__(self):
        import cv2
        import mediapipe as mp

        self.cv2 = cv2
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        # red dots + white lines, like the video
        self.lm_spec = self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=3)
        self.conn_spec = self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam could not be opened (try another CAM_INDEX).")

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def fingers_up(self, lm):
        """Return list of 5 booleans [thumb, index, middle, ring, pinky]."""
        wrist = lm[0]
        up = []
        # thumb: tip farther from the pinky base than the IP joint is
        up.append(self._dist(lm[4], lm[17]) > self._dist(lm[3], lm[17]) * 1.15)
        for tip, pip in zip(self.TIPS, self.PIPS):
            up.append(self._dist(lm[tip], wrist) > self._dist(lm[pip], wrist) * 1.08)
        return up

    def classify(self, lm):
        t, i, m, r, p = self.fingers_up(lm)
        n_up = sum([i, m, r, p])
        if n_up >= 4 and t:
            return MODE_FREE
        if n_up == 0:
            return MODE_HEART
        if i and m and not r and not p:
            return MODE_TEXT
        if i and not m and not r and not p:
            return MODE_SATURN
        if n_up >= 4:
            return MODE_FREE
        return None  # unknown gesture -> keep current mode

    def read(self, current_mode):
        """Returns (mode, hand_xy_norm or None, preview_frame_bgr, running)."""
        cv2 = self.cv2
        ok, frame = self.cap.read()
        if not ok:
            return current_mode, None, None, True
        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        mode = MODE_FREE
        hand_xy = None
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame, hl, self.mp_hands.HAND_CONNECTIONS, self.lm_spec, self.conn_spec
            )
            lm = hl.landmark
            g = self.classify(lm)
            mode = g if g is not None else current_mode
            # palm centre = average of wrist and the four finger bases
            idx = [0, 5, 9, 13, 17]
            hand_xy = (
                sum(lm[k].x for k in idx) / len(idx),
                sum(lm[k].y for k in idx) / len(idx),
            )
        cv2.putText(
            frame, MODE_LABEL[mode], (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2
        )
        return mode, hand_xy, frame, True

    def show(self, frame):
        self.cv2.imshow("Hand Sensor Monitor", frame)
        k = self.cv2.waitKey(1) & 0xFF
        return k not in (27, ord("q"))

    def close(self):
        self.cap.release()
        self.cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    demo = "--demo" in sys.argv
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Space Gesture Controller - Mac Optimized")
    clock = pygame.time.Clock()
    ps = ParticleSystem()

    tracker = None if demo else HandTracker()
    mode = MODE_FREE
    demo_cycle = [MODE_FREE, MODE_HEART, MODE_TEXT, MODE_SATURN]
    demo_start = time.time()
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        if demo:
            k = int((time.time() - demo_start) // 4) % len(demo_cycle)
            mode = demo_cycle[k]
            ph = (time.time() - demo_start) * 0.6
            hand = (0.5 + 0.28 * math.sin(ph), 0.5 + 0.18 * math.cos(ph * 0.8))
        else:
            mode, hand, frame, ok = tracker.read(mode)
            if frame is not None and not tracker.show(frame):
                running = False

        ps.set_mode(mode)
        if hand is not None:
            ps.set_center(hand[0] * WIN_W, hand[1] * WIN_H)
        else:
            ps.set_center(WIN_W / 2, WIN_H / 2)
        ps.update(dt * 60 / 60)
        ps.draw(screen)
        pygame.display.flip()

    if tracker:
        tracker.close()
    pygame.quit()


if __name__ == "__main__":
    main()
