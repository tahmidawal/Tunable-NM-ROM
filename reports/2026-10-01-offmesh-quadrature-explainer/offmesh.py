"""Off-mesh quadrature explainer: what Hari did, and what it did for our Burgers ROM.

Render each scene with manim (conda-forge build), then concatenate with ffmpeg; see build.sh.
Numbers come from vidnums.py, which checks them against the generated reports.
"""
import numpy as np
from manim import *

import hari_quadrature as hq
from vidnums import v, check

check()

config.background_color = "#0e1014"
SERIF = "Latin Modern Roman"
SANS = "DejaVu Sans"

BANK = BLUE_C        # frozen bank / ROM
MESH = RED_C         # mesh, upwind, mesh-bound rules
OFF = YELLOW_C       # off-mesh quadrature points
SOLVE = GREEN_C      # what is solved / results
DIM = GREY_B


# ---------------------------------------------------------------- helpers
def T(s, size=30, color=WHITE, font=SERIF, **kw):
    return Text(s, font=font, font_size=size, color=color, **kw)


def colormap(a):
    """Dark-blue -> blue -> teal -> yellow, a in [0,1]."""
    stops = np.array([[14, 16, 40], [40, 80, 170], [40, 170, 170], [250, 220, 90]], float)
    xs = np.linspace(0, 1, len(stops))
    a = np.clip(a, 0, 1)
    out = np.stack([np.interp(a, xs, stops[:, k]) for k in range(3)], -1)
    return out.astype(np.uint8)


def diverging(a):
    """a in [-1,1]: blue - dark - orange."""
    a = np.clip(a, -1, 1)
    neg = np.array([60, 120, 230.]); pos = np.array([240, 150, 50.]); mid = np.array([20, 22, 30.])
    w = np.abs(a)[..., None]
    col = np.where(a[..., None] >= 0, mid * (1 - w) + pos * w, mid * (1 - w) + neg * w)
    return col.astype(np.uint8)


def img(field, side, cmap=colormap):
    rgb = cmap(field)
    im = ImageMobject(rgb[::-1])  # row 0 at the bottom
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
    im.height = side
    return im


G = np.linspace(0, 1, 120)
XX, YY = np.meshgrid(G, G)
DIRI = np.sin(np.pi * XX) * np.sin(np.pi * YY)


def bump(t, x=XX, y=YY):
    """Illustrative Burgers-like bump: moves along the diagonal and steepens at its front."""
    s = (x + y) / 2
    c = 0.32 + 0.30 * t
    wb, wf = 0.14, 0.14 - 0.115 * t
    w = np.where(s > c, wf, wb)
    perp = (x - y) / 2
    return np.exp(-((s - c) / w) ** 2 - (perp / 0.16) ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y) ** 0.3 \
        * np.sin(np.pi * y) ** 0.0


def basis(seed):
    r = np.random.default_rng(seed)
    f = np.zeros_like(XX)
    for _ in range(4):
        a, b = r.integers(1, 5, 2)
        f += r.normal() * np.sin(a * np.pi * XX + r.uniform(0, 3)) * np.sin(b * np.pi * YY + r.uniform(0, 3))
    f *= DIRI
    return f / np.abs(f).max()


def test_fn(a, b):
    return np.sin(a * np.pi * XX) * np.sin(b * np.pi * YY)


class Base(Scene):
    caption = None

    def header(self, s):
        h = T(s, 34, color=GREY_A).to_corner(UL, buff=0.45)
        line = Line(LEFT, RIGHT, stroke_width=1.5, color=GREY_D).set_width(h.width).next_to(h, DOWN, 0.12, aligned_edge=LEFT)
        self.play(FadeIn(h, shift=0.2 * RIGHT), Create(line), run_time=0.8)
        self.hdr = VGroup(h, line)
        return self.hdr

    def say(self, s, extra=0.0, size=28):
        new = T("\n".join(wrap(s, 82)), size, color=GREY_A, font=SANS, line_spacing=0.8)
        if new.width > 13.4:
            new.scale_to_fit_width(13.4)
        new.to_edge(DOWN, buff=0.35)
        if self.caption is None:
            self.play(FadeIn(new, shift=0.1 * UP), run_time=0.5)
        else:
            self.play(FadeOut(self.caption, shift=0.1 * UP), FadeIn(new, shift=0.1 * UP), run_time=0.5)
        self.caption = new
        self.wait(0.7 + len(s.split()) / 2.7 + extra)

    def unsay(self):
        if self.caption is not None:
            self.play(FadeOut(self.caption), run_time=0.4)
            self.caption = None

    def clear_all(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)
        self.caption = None


def grid_lines(n, side, color=MESH, width=1.0, opacity=0.6):
    g = VGroup()
    for i in range(n + 1):
        x = -side / 2 + side * i / n
        g.add(Line([x, -side / 2, 0], [x, side / 2, 0]))
        g.add(Line([-side / 2, x, 0], [side / 2, x, 0]))
    return g.set_stroke(color, width, opacity)


def unit_to(square, pts):
    """Map points in [0,1]^2 to scene coordinates inside a Square mobject."""
    side = square.width
    c = square.get_center()
    return [c + np.array([(p[0] - 0.5) * side, (p[1] - 0.5) * side, 0]) for p in pts]


# ================================================================ 0 · intro
class S0Intro(Base):
    def construct(self):
        t1 = T("Leaving the mesh", 64)
        t2 = T("Hari's off-mesh quadrature, step by step,", 32, color=GREY_A)
        t3 = T("on our own Burgers reduced-order model", 32, color=GREY_A)
        g = VGroup(t1, t2, t3).arrange(DOWN, buff=0.3).shift(0.8 * UP)
        self.play(Write(t1), run_time=1.5)
        self.play(FadeIn(t2, shift=0.2 * UP), FadeIn(t3, shift=0.2 * UP))
        self.wait(1)
        qs = VGroup(
            T("1.  What does our ROM compute every time step?", 30),
            T("2.  Which part of it is expensive, and how did we handle it?", 30),
            T("3.  What did Hari change?", 30),
            T("4.  What did that change buy us, in 3D and in 2D?", 30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).next_to(g, DOWN, 0.8)
        for q in qs:
            self.play(FadeIn(q, shift=0.2 * RIGHT), run_time=0.6)
        self.wait(3)
        self.clear_all()


# ================================================================ 1 · our ROM
class S1OurROM(Base):
    def construct(self):
        self.header("1 · What our ROM does every time step")
        pde = MathTex(r"u_t", r"+", r"u\,(u_x+u_y)", r"=", r"\nu\,\Delta u", font_size=54).shift(1.9 * UP)
        pde[2].set_color(OFF); pde[4].set_color(BANK)
        la = T("advection (nonlinear)", 24, color=OFF).next_to(pde[2], DOWN, 0.3).shift(0.9 * LEFT)
        lb = T("diffusion (linear)", 24, color=BANK).next_to(pde[4], DOWN, 0.3).shift(0.7 * RIGHT)
        self.play(Write(pde))
        self.play(FadeIn(la), FadeIn(lb))
        self.say("Burgers' equation: a bump of fluid moves and steepens (advection) while it smooths out (diffusion).")

        side = 3.0
        tr = ValueTracker(0)
        field = always_redraw(lambda: img(bump(tr.get_value()), side).move_to(0.75 * DOWN))
        frame = Square(side, stroke_color=GREY_C, stroke_width=2).move_to(0.75 * DOWN)
        tag = T("illustration", 18, color=GREY_C).next_to(frame, DOWN, 0.1)
        self.play(FadeIn(field), Create(frame), FadeIn(tag))
        self.play(tr.animate.set_value(1), run_time=3.5, rate_func=smooth)
        self.play(tr.animate.set_value(0.4), run_time=1.5)
        mesh = grid_lines(24, side).move_to(frame)
        self.play(Create(mesh), run_time=1.5)
        self.say("A full-order solver stores one unknown per mesh node: 256² = 65,536 in 2D, and 256³ ≈ 16.8 million in 3D.", 1)
        self.play(FadeOut(mesh))

        # --- the bank
        field.clear_updaters()
        still = img(bump(0.4), side).move_to(field)
        self.remove(field); self.add(still); field = still
        self.play(FadeOut(VGroup(pde, la, lb)), Group(field, frame, tag).animate.scale(0.62).to_edge(LEFT, buff=0.7).shift(0.6 * UP))
        net = RoundedRectangle(corner_radius=0.15, width=2.2, height=1.4, color=BANK, fill_opacity=0.15)
        net_t = MathTex(r"G", font_size=60, color=BANK).move_to(net)
        net_g = VGroup(net, net_t).move_to(1.6 * UP + 1.2 * RIGHT)
        xin = MathTex(r"x=(x,y)", font_size=40).next_to(net_g, LEFT, 0.9)
        out = MathTex(r"G(x)\in\mathbb{R}^{R'}", font_size=40).next_to(net_g, RIGHT, 0.9)
        a1 = Arrow(xin.get_right(), net_g.get_left(), buff=0.1)
        a2 = Arrow(net_g.get_right(), out.get_left(), buff=0.1)
        self.play(FadeIn(net_g), Write(xin), GrowArrow(a1))
        self.play(GrowArrow(a2), Write(out))
        self.say("Our ROM's bank G is a neural network of the coordinates: give it any point x, it returns R′ numbers.")

        thumbs = Group()
        cs = ["c_1", "c_2", "c_3", r"\cdots", "c_{R'}"]
        for i, c in enumerate(cs):
            if c == r"\cdots":
                thumbs.add(MathTex(r"+\cdots+", font_size=36))
                continue
            im = img((basis(i) + 1) / 2, 1.0, cmap=lambda a: diverging(2 * a - 1))
            lab = MathTex(c, font_size=32).next_to(im, LEFT, 0.08)
            plus = MathTex("+", font_size=36) if i > 0 and cs[i - 1] != r"\cdots" else Mobject()
            thumbs.add(Group(plus, lab, im).arrange(RIGHT, buff=0.1) if i > 0 and cs[i - 1] != r"\cdots" else Group(lab, im).arrange(RIGHT, buff=0.08))
        thumbs.arrange(RIGHT, buff=0.18)
        eq = MathTex("=", font_size=40)
        res = img(bump(0.4), 1.0)
        row = Group(thumbs, eq, res).arrange(RIGHT, buff=0.2).move_to(0.1 * DOWN + 1.3 * RIGHT)
        if row.width > 8.6:
            row.scale_to_fit_width(8.6).move_to(0.1 * DOWN + 1.3 * RIGHT)
        formula = MathTex(r"u(x)", r"=", r"\sum_{r=1}^{R'} c_r\,G_r(x)", r"=", r"G(x)\,c", font_size=44).next_to(row, DOWN, 0.45)
        formula[4].set_color(BANK)
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in thumbs], lag_ratio=0.15), run_time=1.5)
        self.play(FadeIn(eq), FadeIn(res))
        self.play(Write(formula))
        self.say("The flow field is a weighted sum of the bank's outputs. The network is frozen; each step we only solve for the R′ weights c (R′ ≤ 512).", 0.5)

        self.play(FadeOut(Group(net_g, xin, out, a1, a2, thumbs, eq, res, formula, field, frame, tag)))
        tests = Group(*[Group(img((test_fn(a, b) + 1) / 2, 1.3, cmap=lambda z: diverging(2 * z - 1)),
                              MathTex(rf"\psi_{{{a}{b}}}", font_size=34)).arrange(DOWN, buff=0.15)
                        for a, b in [(1, 1), (2, 3), (5, 2), (7, 7)]]).arrange(RIGHT, buff=0.5).shift(1.75 * UP)
        tdef = MathTex(r"\psi_{ab}(x)=\sin(a\pi x)\,\sin(b\pi y)", font_size=40).next_to(tests, DOWN, 0.35)
        self.play(LaggedStart(*[FadeIn(t) for t in tests], lag_ratio=0.2), Write(tdef))
        self.say("To pick c, each step asks the equation to balance when tested against M sine waves ψ_ab (M ≈ 4R′).")

        res_eq = MathTex(r"r(c)", r"=", r"\underbrace{A c - p + \Delta t\,\nu\Lambda A c}_{\text{linear: small matrices, built once}}",
                         r"+\;\Delta t\,", r"\underbrace{N(c)}_{\text{tested advection}}", font_size=40).shift(0.55 * DOWN)
        res_eq[2].set_color(BANK); res_eq[4].set_color(OFF)
        nab = MathTex(r"N_{ab}(c)=\int_\Omega \psi_{ab}\;u\,(u_x+u_y)\,dx", font_size=42, color=OFF).next_to(res_eq, DOWN, 0.3)
        self.play(Write(res_eq))
        self.say("The linear terms collapse, offline, into small matrices of size about M × R′. Cheap.")
        self.play(Write(nab))
        self.play(Indicate(nab, color=OFF, scale_factor=1.08))
        self.say("The advection term N(c) is quadratic in c. It must be re-evaluated inside every Levenberg–Marquardt iteration of every time step.", 0.5)
        self.clear_all()


# ================================================================ 2 · bottleneck
class S2Bottleneck(Base):
    def construct(self):
        self.header("2 · The bottleneck, and how we handled it before")
        side = 3.8
        frame = Square(side, stroke_color=GREY_C).shift(2.8 * LEFT + 0.3 * DOWN)
        n = 20
        pts = [[(i + 0.5) / n, (j + 0.5) / n] for i in range(n) for j in range(n)]
        dots = VGroup(*[Dot(p, radius=0.035, color=MESH) for p in unit_to(frame, pts)])
        self.play(Create(frame), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.002), run_time=1.5)
        msum = MathTex(r"N_{ab}(c)\;\approx\;h^2\!\!\sum_{i\,\in\,\text{mesh}}\psi_{ab}(x_i)\,u_i\,(D^{\mathrm{up}}u)_i",
                       font_size=40).shift(2.6 * RIGHT + 1.6 * UP)
        self.play(Write(msum))
        self.say("On the mesh, the integral becomes a sum over every node, using the solver's upwind difference D^up for the derivative.")
        self.play(LaggedStart(*[d.animate.set_color(OFF).scale(1.6) for d in dots], lag_ratio=0.003), run_time=2.5)
        self.play(*[d.animate.set_color(MESH).scale(1 / 1.6) for d in dots], run_time=0.4)

        # cost growth bars
        ax_lbl = T("work per evaluation", 22, color=GREY_B)
        bars = VGroup()
        for k, (lab, h) in enumerate([("64²", 0.12), ("128²", 0.48), ("256²", 1.92)]):
            b = Rectangle(width=0.7, height=h, fill_color=MESH, fill_opacity=0.8, stroke_width=0)
            l = T(lab, 20, color=GREY_B)
            bars.add(VGroup(b, l))
        for k, bb in enumerate(bars):
            bb[0].move_to([1.6 + 1.2 * k, -1.9 + bb[0].height / 2, 0])
            bb[1].next_to(bb[0], DOWN, 0.1)
        ax_lbl.next_to(bars, UP, 0.3).shift(0.3 * UP)
        self.play(FadeIn(ax_lbl), LaggedStart(*[GrowFromEdge(bb[0], DOWN) for bb in bars], lag_ratio=0.4),
                  LaggedStart(*[FadeIn(bb[1]) for bb in bars], lag_ratio=0.4))
        self.say("The work grows with the number of nodes: ×4 per refinement in 2D, ×8 in 3D. Done every iteration, the ROM is no faster than the full solver.", 0.5)
        self.play(FadeOut(VGroup(bars, ax_lbl, msum)))

        # three previous fixes
        self.play(VGroup(frame, dots).animate.scale(0.55).move_to(4.6 * LEFT + 0.9 * UP))
        p1 = VGroup(frame, dots)
        sub = [d for k, d in enumerate(dots) if (k // n) % 3 == 1 and (k % n) % 3 == 1]
        rest = [d for d in dots if not any(d is q for q in sub)]
        self.play(*[d.animate.set_opacity(0.25) for d in rest], *[d.animate.set_color(OFF).scale(2.0) for d in sub], run_time=1)
        l1 = VGroup(T("Mesh sub-lattice", 26, color=WHITE), T("2D: 63² of the mesh nodes", 18, color=GREY_B),
                    T("weight = cells each stands for", 18, color=GREY_B)).arrange(DOWN, buff=0.08).next_to(p1, DOWN, 0.3)
        self.play(FadeIn(l1))

        frame2 = frame.copy().move_to(0.9 * UP)
        r = np.random.default_rng(3)
        pick = r.choice(len(pts), 34, replace=False)
        eqd = VGroup(*[Dot(unit_to(frame2, [pts[k]])[0], radius=0.02 + 0.06 * r.uniform(), color=OFF) for k in pick])
        faint = VGroup(*[Dot(unit_to(frame2, [p])[0], radius=0.02, color=MESH).set_opacity(0.25) for p in pts])
        l2 = VGroup(T("Empirical quadrature (EQ)", 26), T("2D head: nodes and weights", 18, color=GREY_B),
                    T("fitted by NNLS to training states", 18, color=GREY_B)).arrange(DOWN, buff=0.08).next_to(frame2, DOWN, 0.3)
        self.play(FadeIn(frame2), FadeIn(faint), LaggedStart(*[GrowFromCenter(d) for d in eqd], lag_ratio=0.05), FadeIn(l2))

        cube = VGroup(*[Square(1.5, fill_color=MESH, fill_opacity=0.25 + 0.15 * k, stroke_color=MESH, stroke_width=1.5)
                        .shift(k * 0.18 * (UP + RIGHT)) for k in range(5)]).scale(0.8).move_to(4.6 * RIGHT + 0.8 * UP)
        tl = MathTex(r"N(c)=\tfrac12\,c^{\top}\,T\,c", font_size=34).next_to(cube, UP, 0.25)
        l3 = VGroup(T("Quadratic tensor", 26), T("3D: T built from the full mesh sum", 18, color=GREY_B),
                    T(f"M·R′² numbers = {v('mem', 0)}", 18, color=GREY_B)).arrange(DOWN, buff=0.08).next_to(cube, DOWN, 0.3)
        self.play(FadeIn(cube, lag_ratio=0.2), Write(tl), FadeIn(l3))
        self.say("Before Hari's work we had three fixes: a fixed subset of mesh nodes, an empirical quadrature fitted to training states, and, in 3D, a precomputed tensor.", 0.5)
        box = SurroundingRectangle(VGroup(p1, frame2, cube, l1, l2, l3, tl), color=MESH, buff=0.2)
        self.play(Create(box))
        self.say("Note: all three aim to reproduce the mesh sum, upwind difference included. That detail matters later.", 1)
        self.clear_all()


# ================================================================ 3 · Hari's idea
class S3Idea(Base):
    def construct(self):
        self.header("3 · Hari's idea: leave the mesh")
        side = 3.8
        frame = Square(side, stroke_color=GREY_C).shift(4.2 * LEFT + 0.35 * UP)
        fld = img(bump(0.4), side).move_to(frame)
        mesh = grid_lines(16, side, opacity=0.35).move_to(frame)
        self.play(FadeIn(fld), Create(frame), Create(mesh))

        # moving probe: u and gradient at an arbitrary point
        tt = ValueTracker(0)

        def probe_xy(s):
            a = 2 * np.pi * s
            return np.array([0.44 + 0.13 * np.cos(a) - 0.06 * np.sin(a), 0.44 + 0.13 * np.cos(a) + 0.06 * np.sin(a)]) + 0.07 * np.array([np.sin(2 * a), -np.sin(2 * a)])

        def u_at(p):
            return float(bump(0.4, np.array([[p[0]]]), np.array([[p[1]]]))[0, 0])

        def grad_at(p, e=1e-4):
            return np.array([(u_at(p + [e, 0]) - u_at(p - [e, 0])) / (2 * e), (u_at(p + [0, e]) - u_at(p - [0, e])) / (2 * e)])

        dot = always_redraw(lambda: Dot(unit_to(frame, [probe_xy(tt.get_value())])[0], radius=0.08, color=WHITE))
        arrow = always_redraw(lambda: Arrow(
            unit_to(frame, [probe_xy(tt.get_value())])[0],
            unit_to(frame, [probe_xy(tt.get_value())])[0] + 0.08 * np.append(grad_at(probe_xy(tt.get_value())), 0),
            buff=0, color=OFF, stroke_width=4, max_tip_length_to_length_ratio=0.3))
        read = always_redraw(lambda: VGroup(
            MathTex(rf"x=({probe_xy(tt.get_value())[0]:.3f},\,{probe_xy(tt.get_value())[1]:.3f})", font_size=36),
            MathTex(rf"u(x)=G(x)\,c={u_at(probe_xy(tt.get_value())):.3f}", font_size=36),
            MathTex(r"\nabla u(x)=\nabla G(x)\,c", font_size=36, color=OFF),
        ).arrange(DOWN, aligned_edge=LEFT).move_to(2.2 * RIGHT + 1.0 * UP))
        self.add(dot, arrow, read)
        self.play(tt.animate.set_value(1), run_time=6, rate_func=linear)
        self.say("The key fact: our bank is a function of continuous coordinates. We can read u and its exact gradient (by autodiff) at any point, not only at mesh nodes.", 0.5)
        for m in (dot, arrow, read):
            m.clear_updaters()
        self.play(FadeOut(VGroup(dot, arrow, read)))

        e1 = MathTex(r"N_{ab}(c)", r"=", r"\int_\Omega \psi_{ab}\,u\,(u_x+u_y)\,dx", font_size=40).move_to(1.6 * RIGHT + 1.9 * UP)
        e2 = MathTex(r"\approx", r"\sum_{q=1}^{m} w_q\,\psi_{ab}(x_q)\,u(x_q)\,(u_x+u_y)(x_q)", font_size=40).next_to(e1, DOWN, 0.35, aligned_edge=LEFT).shift(0.95 * RIGHT)
        e2[1].set_color(OFF)
        self.play(Write(e1))
        self.say("So treat the advection term as the true integral over the domain, and approximate it with a classical quadrature rule:")
        self.play(Write(e2))
        X, W = hq.gauss_tensor(12)
        qd = VGroup(*[Dot(p, radius=0.015 + 0.55 * np.sqrt(w), color=OFF) for p, w in zip(unit_to(frame, X), W)])
        self.play(mesh.animate.set_stroke(opacity=0.12), LaggedStart(*[GrowFromCenter(d) for d in qd], lag_ratio=0.01), run_time=2)
        self.say("m fixed points x_q with weights w_q (here a 12 × 12 Gauss rule; dot size = weight). They do not sit on mesh nodes.")

        # refine the mesh, points stay put
        lab = always_redraw(lambda: T("", 1))
        nlab = VGroup(T("mesh:", 26, color=MESH), T("16 × 16", 26, color=MESH)).arrange(RIGHT).next_to(frame, DOWN, 0.25)
        mlab = VGroup(T("points:", 26, color=OFF), T(f"m = {len(X)}", 26, color=OFF)).arrange(RIGHT).next_to(nlab, DOWN, 0.12)
        self.play(FadeIn(nlab), FadeIn(mlab), mesh.animate.set_stroke(opacity=0.35))
        for nn in (32, 64):
            new = grid_lines(nn, side, opacity=0.35, width=0.8).move_to(frame)
            nl = T(f"{nn} × {nn}", 26, color=MESH).move_to(nlab[1], aligned_edge=LEFT)
            self.play(Transform(mesh, new), Transform(nlab[1], nl), Indicate(qd, scale_factor=1.0, color=OFF), run_time=1.3)
            self.wait(0.4)
        self.say("Refine the mesh and nothing about the points changes. The cost of the term depends on m, not on the mesh size.", 0.5)

        self.play(FadeOut(VGroup(e1, e2)))
        off = VGroup(
            T("Offline, once:", 26, color=GREY_A),
            MathTex(r"B = G(x_q)\in\mathbb{R}^{m\times R'},\quad D=\partial G(x_q)\in\mathbb{R}^{m\times R'}", font_size=34),
            T("Online, every iteration:", 26, color=GREY_A),
            MathTex(r"N(c)\;=\;P^{\top}\big[(Bc)\odot(Dc)\big]", font_size=42, color=OFF),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(2.3 * RIGHT + 0.9 * UP)
        self.play(FadeIn(off, lag_ratio=0.3), run_time=2)
        self.say("In practice: decode the bank and its derivative at the points once. Online, the term is two small matrix–vector products and one matrix multiply with the tests.", 0.5)
        self.say("Nothing is fitted to data, and nothing has to be recomputed when the mesh changes. This is Hari's proposal.", 1)
        self.clear_all()


# ================================================================ 4 · which points
class S4Points(Base):
    def construct(self):
        self.header("4 · Which points? Not all rules are equal")
        rules = [
            ("Gauss tensor", hq.gauss_tensor(14)),
            ("Fibonacci lattice", hq.fibonacci_lattice(13)),
            ("Sobol (scrambled)", hq.sobol(256)),
            ("Smolyak sparse grid", hq.smolyak(6)),
        ]
        panels = VGroup()
        for name, (X, W) in rules:
            sq = Square(2.6, stroke_color=GREY_C, stroke_width=1.5)
            ds = VGroup(*[Dot(p, radius=0.022, color=OFF) for p in unit_to(sq, X)])
            lab = T(name, 22).next_to(sq, UP, 0.15)
            cnt = T(f"m = {len(X)}", 18, color=GREY_B).next_to(sq, DOWN, 0.1)
            panels.add(VGroup(sq, ds, lab, cnt))
        panels.arrange(RIGHT, buff=0.45).shift(0.5 * UP)
        for p in panels:
            self.play(FadeIn(p[0]), FadeIn(p[2]), LaggedStart(*[GrowFromCenter(d) for d in p[1]], lag_ratio=0.004), FadeIn(p[3]), run_time=1.1)
        self.say("Hari compared classical rules: a tensor Gauss grid, a rank-1 (Fibonacci) lattice, a scrambled Sobol sequence, and a Smolyak sparse grid.")

        # why smolyak fails
        smol = panels[3]
        hi = img((test_fn(9, 9) + 1) / 2, 2.6, cmap=lambda z: diverging(2 * z - 1)).move_to(smol[0]).set_opacity(0.75)
        self.play(FadeIn(hi))
        self.bring_to_front(smol[1])
        self.say("The sparse grid keeps fine detail along each axis but drops 'mixed' detail in both directions at once. Our tests ψ_ab with large a and b are exactly that kind of function.", 0.5)
        self.play(FadeOut(hi))
        self.say("The integrand is smooth and vanishes at the walls, so the Gauss and lattice rules converge faster than any power of m.")
        self.play(FadeOut(panels))

        # convergence on OUR 3D model
        ax = Axes(x_range=[3.3, 4.7, 0.5], y_range=[0, 7, 1], x_length=7.5, y_length=3.7,
                  axis_config={"color": GREY_C, "include_ticks": False}, tips=False).shift(0.9 * LEFT + 0.35 * UP)
        _c2p = ax.c2p
        c2p = lambda x, y: _c2p(x, y + 5)
        xl = VGroup(*[MathTex(s, font_size=26).next_to(c2p(np.log10(m), -5), DOWN, 0.15)
                      for s, m in [("4096", 4096), ("8192", 8192), ("16384", 16384), ("32768", 32768)]])
        yl = VGroup(*[MathTex(rf"10^{{{k}}}", font_size=26).next_to(c2p(3.3, k), LEFT, 0.12) for k in range(-5, 3)])
        xt = T("number of points m", 22, color=GREY_B).next_to(ax, DOWN, 0.5)
        yt = T("worst quadrature error ρ", 22, color=GREY_B).rotate(PI / 2).next_to(yl, LEFT, 0.2)
        title = T("Our Burgers 3D model, R′ = 512, 64³: worst ρ over reached states", 24, color=GREY_A).next_to(ax, UP, 0.3)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), FadeIn(title))

        def series(ms, vals, col, name):
            p = [c2p(np.log10(m), np.log10(float(x))) for m, x in zip(ms, vals)]
            line = VMobject(color=col, stroke_width=4).set_points_as_corners(p) if len(p) > 1 else VGroup()
            dts = VGroup(*[Dot(q, color=col, radius=0.07) for q in p])
            lab = T(name, 22, color=col).next_to(p[-1], RIGHT, 0.15)
            if name == "Sobol":
                lab.next_to(p[-1], UR, 0.08)
            return VGroup(line, dts, lab)

        lat = series([4096, 8192, 16384, 32768], SRC_vals("rho3d_lat"), OFF, "CBC lattice")
        gl = series([4096, 13824, 32768], SRC_vals("rho3d_gl"), BANK, "Gauss 16³ / 24³ / 32³")
        sob = series([16384], SRC_vals("rho3d_sob"), GREY_B, "Sobol")
        smo = series([2559], SRC_vals("rho3d_smol"), MESH, "Smolyak (2559 pts)")
        bar = DashedLine(c2p(3.3, np.log10(0.116)), c2p(4.7, np.log10(0.116)), color=GREY_A)
        barl = T(f"certificate bar {v('bar')}", 18, color=GREY_A).next_to(bar, DOWN, 0.08).align_to(bar, RIGHT)
        self.play(Create(bar), FadeIn(barl))
        for s in (smo, sob, gl, lat):
            anims = [FadeIn(s[1]), FadeIn(s[2])]
            if isinstance(s[0], VMobject) and s[0].has_points():
                anims.append(Create(s[0]))
            self.play(*anims, run_time=1.2)
        self.say(f"Measured on our own 3D model: at 4096 points the lattice is at {fmt(v('rho3d_lat',0))}, Gauss 16³ at {fmt(v('rho3d_gl',0))}. Smolyak is far off (≈ 43), Sobol stalls near 0.15.", 0.5)
        self.say("In 3D the lattice beats Gauss at equal point count everywhere. Lattice points cost one point each in any dimension; a Gauss grid's count is cubed.", 0.5)
        self.clear_all()


def SRC_vals(key):
    from vidnums import SRC
    return SRC[key][0]


def fmt(s):
    m, e = s.split("e")
    return f"{float(m):g}×10^{int(e)}" if int(e) < -2 else f"{float(s):g}"


# ================================================================ 5 · twist
class S5Twist(Base):
    def construct(self):
        self.header("5 · The twist: which answer is each rule matching?")
        nl = NumberLine(x_range=[0, 10, 1], length=10, include_ticks=False, color=GREY_D).shift(1.4 * UP)
        cont = Dot(nl.n2p(2), radius=0.12, color=SOLVE)
        mesh = Dot(nl.n2p(7.5), radius=0.12, color=MESH)
        lc = VGroup(T("true integral", 24, color=SOLVE), T("(continuum)", 20, color=GREY_B)).arrange(DOWN, buff=0.05).next_to(cont, UP, 0.25)
        lm = VGroup(T("mesh sum", 24, color=MESH), T("(upwind difference)", 20, color=GREY_B)).arrange(DOWN, buff=0.05).next_to(mesh, UP, 0.25)
        gap = BraceBetweenPoints(cont.get_center(), mesh.get_center(), DOWN)
        gl = MathTex(r"O(h)\ \text{gap}", font_size=34).next_to(gap, DOWN, 0.1)
        self.play(Create(nl), FadeIn(cont), FadeIn(lc))
        self.play(FadeIn(mesh), FadeIn(lm))
        self.play(GrowFromCenter(gap), Write(gl))
        self.say("The mesh sum is not the true integral: the upwind difference is only first-order accurate, so the two differ by an amount proportional to the mesh spacing h.", 0.5)

        mr = VGroup(T("tensor, sub-lattice, EQ", 22, color=MESH)).next_to(mesh, DOWN, 1.3)
        orr = VGroup(T("off-mesh Gauss / lattice", 22, color=OFF)).next_to(cont, DOWN, 1.3)
        a1 = Arrow(mr.get_top(), mesh.get_bottom(), color=MESH, buff=0.15)
        a2 = Arrow(orr.get_top(), cont.get_bottom(), color=OFF, buff=0.15)
        self.play(FadeIn(mr), GrowArrow(a1))
        self.play(FadeIn(orr), GrowArrow(a2))
        self.say("Our earlier rules copy the mesh sum, error and all. Hari's off-mesh rules aim at the true integral instead.")

        gaps = SRC_vals("gap")
        bars = VGroup()
        for k, (lab, val) in enumerate(zip(["64³", "128³", "256³"], gaps)):
            h = float(val) * 12
            b = Rectangle(width=0.8, height=h, fill_color=MESH, fill_opacity=0.85, stroke_width=0).move_to([-1.5 + 1.5 * k, -2.0 + h / 2, 0])
            bars.add(VGroup(b, T(lab, 20, color=GREY_B).next_to(b, DOWN, 0.1), T(fmt(val), 20).next_to(b, UP, 0.08)))
        bt = T("measured gap ρ, mesh sum vs true integral (our 3D model)", 22, color=GREY_A).next_to(bars, UP, 0.35).shift(0.4 * UP)
        self.play(FadeOut(VGroup(mr, orr, a1, a2)), VGroup(nl, cont, mesh, lc, lm, gap, gl).animate.scale(0.7).shift(0.6 * UP))
        VGroup(bt, bars).shift(0.3 * DOWN)
        self.play(FadeIn(bt), LaggedStart(*[FadeIn(b, shift=0.2 * UP) for b in bars], lag_ratio=0.3))
        self.say("On our 3D model the gap roughly halves each time the mesh is refined, as a first-order error should.", 0.5)
        self.clear_all()

        # error vs mesh
        self.header("5 · Consequence: the error stops depending on the mesh")
        ax = Axes(x_range=[0, 2, 1], y_range=[0, 12, 2], x_length=7, y_length=3.8, tips=False,
                  axis_config={"color": GREY_C}).shift(1.4 * LEFT + 0.35 * UP)
        xl = VGroup(*[T(s, 22, color=GREY_B).next_to(ax.c2p(i, 0), DOWN, 0.15) for i, s in enumerate(["64³", "128³", "256³"])])
        yl = VGroup(*[T(f"{k}%", 20, color=GREY_B).next_to(ax.c2p(0, k), LEFT, 0.12) for k in range(0, 13, 2)])
        tl = T("worst error over 32 held-out cases, vs a 513³ refined reference (R′ = 512)", 22, color=GREY_A).next_to(ax, UP, 0.3)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(tl))

        def ln(vals, col, name):
            p = [ax.c2p(i, float(x)) for i, x in enumerate(vals)]
            l = VMobject(color=col, stroke_width=5).set_points_as_corners(p)
            d = VGroup(*[Dot(q, color=col, radius=0.07) for q in p])
            where = {MESH: UR, GREY_B: DR, OFF: UP}[col]
            nums = VGroup(*[T(f"{x}%", 18, color=col).next_to(q, where, 0.08) for q, x in zip(p, vals)])
            lab = T(name, 22, color=col).next_to(p[-1], RIGHT, 0.9)
            return VGroup(l, d, nums, lab)

        ten = ln(SRC_vals("err_tensor"), MESH, "tensor (before)")
        fom = ln(SRC_vals("err_fom"), GREY_B, "best full solver, same mesh")
        off = ln([v("err_off")] * 3, OFF, "off-mesh (after)")
        fom[3].shift(0.45 * DOWN); ten[3].shift(0.45 * UP)
        off[2][1:].set_opacity(0)
        for l in (ten, fom, off):
            self.play(Create(l[0]), FadeIn(l[1]), FadeIn(l[2]), FadeIn(l[3]), run_time=1.5)
            self.wait(0.3)
        self.say(f"The tensor inherits the mesh's error and only improves as the mesh is refined ({v('err_tensor',0)}% → {v('err_tensor',2)}%). The off-mesh solve sits at {v('err_off')}% at every mesh.", 1)
        self.say(f"Fair caveats: the reference is itself a first-order solve. The bank saw fields up to {v('bank129')} nodes per axis, so its derivatives carry resolution a coarse mesh lacks. At 256³ the best full solver ({v('err_fom',2)}%) is still more accurate.", 1.5)
        self.clear_all()


# ================================================================ 6 · 3D payoff
class S6Payoff3D(Base):
    def construct(self):
        self.header("6 · What it bought us in 3D")
        sub = T("Burgers 3D, R′ = 512, 32 held-out cases", 22, color=GREY_B).next_to(self.hdr, RIGHT, 0.4)
        self.play(FadeIn(sub))
        ms = SRC_vals("ms")  # off,ten pairs per mesh? order in SRC: 47.9 34.1 52.0 40.6 90.1 76.7 = ten,off,...
        ten_s, off_s = [ms[0], ms[2], ms[4]], [ms[1], ms[3], ms[5]]
        ten = [float(x) for x in ten_s]
        off = [float(x) for x in off_s]
        scale = 0.03
        grp = VGroup()
        for k, lab in enumerate(["64³", "128³", "256³"]):
            x0 = -6.0 + 1.75 * k
            b1 = Rectangle(width=0.6, height=ten[k] * scale, fill_color=MESH, fill_opacity=0.85, stroke_width=0).move_to([x0, -1.9 + ten[k] * scale / 2, 0])
            b2 = Rectangle(width=0.6, height=off[k] * scale, fill_color=OFF, fill_opacity=0.85, stroke_width=0).move_to([x0 + 0.65, -1.9 + off[k] * scale / 2, 0])
            t1 = T(ten_s[k], 17).next_to(b1, UP, 0.06)
            t2 = T(off_s[k], 17).next_to(b2, UP, 0.06)
            l = T(lab, 20, color=GREY_B).next_to(VGroup(b1, b2), DOWN, 0.12)
            grp.add(VGroup(b1, b2, t1, t2, l))
        ttl = T("ms per query", 24, color=GREY_A).next_to(grp, UP, 0.3).shift(0.9 * UP)
        leg = VGroup(VGroup(Square(0.22, fill_color=MESH, fill_opacity=0.85, stroke_width=0), T("tensor", 18)).arrange(RIGHT, buff=0.1),
                     VGroup(Square(0.22, fill_color=OFF, fill_opacity=0.85, stroke_width=0), T("off-mesh (Gauss 24³)", 18)).arrange(RIGHT, buff=0.1)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(ttl, DOWN, 0.15)
        self.play(FadeIn(ttl), FadeIn(leg), LaggedStart(*[FadeIn(g, shift=0.2 * UP) for g in grp], lag_ratio=0.3), run_time=2)
        self.say("Faster in every cell, even though it does more arithmetic: it reads far fewer bytes.")

        mem_t = float(v("mem", 0).split()[0]); mem_o = float(v("mem", 1).split()[0])
        mb1 = Rectangle(width=0.8, height=mem_t * 0.65, fill_color=MESH, fill_opacity=0.85, stroke_width=0).move_to([-0.9, -1.9 + mem_t * 0.325, 0])
        mb2 = Rectangle(width=0.8, height=mem_o * 0.65, fill_color=OFF, fill_opacity=0.85, stroke_width=0).move_to([0.2, -1.9 + mem_o * 0.325, 0])
        mt = VGroup(T(v("mem", 0), 18).next_to(mb1, UP, 0.06), T(v("mem", 1), 18).next_to(mb2, UP, 0.06))
        mtl = VGroup(T("memory for the", 24, color=GREY_A), T("advection term", 24, color=GREY_A)).arrange(DOWN, buff=0.05).move_to([-0.35, 2.0, 0])
        self.play(FadeIn(mtl), GrowFromEdge(mb1, DOWN), GrowFromEdge(mb2, DOWN), FadeIn(mt))
        self.say("And the tensor's gigabytes are gone: the off-mesh rule stores only the bank decoded at its points.")

        sp = VGroup(
            T("speedup over the full solver at 256³", 24, color=GREY_A),
            VGroup(T("same-mesh accuracy rule:", 22, color=GREY_B), T(v("speedup", 0), 30, color=MESH),
                   MathTex(r"\to", font_size=36), T(v("speedup", 1), 30, color=OFF)).arrange(RIGHT, buff=0.2),
            VGroup(T("matched to refined accuracy:", 22, color=GREY_B), T(v("speedup", 0), 30, color=MESH),
                   MathTex(r"\to", font_size=36), T(v("speedup", 2), 30, color=OFF)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        sp.scale_to_fit_width(min(sp.width, 5.4)).move_to([4.0, 0.4, 0])
        self.play(FadeIn(sp, lag_ratio=0.3), run_time=1.5)
        self.say(f"At 256³ the speedup rises from {v('speedup',0)} to {v('speedup',1)}. Because the off-mesh solve is more accurate, matching it needs a stricter, slower full solver: {v('speedup',2)}.", 1)
        self.say("Caveat: at 64³ every reduced solve, old or new, is still slower than the full solver.", 0.5)
        self.clear_all()


# ================================================================ 7 · 2D
class S7Results2D(Base):
    def construct(self):
        self.header("7 · Burgers 2D  (R = 512 bank, 64 held-out test cases)")
        # error at 256^2
        e_off, e_den = float(v("err2d", 0)), float(v("err2d", 1))
        b1 = Rectangle(width=0.9, height=e_den * 0.55, fill_color=MESH, fill_opacity=0.85, stroke_width=0).move_to([-5.4, -2.0 + e_den * 0.275, 0])
        b2 = Rectangle(width=0.9, height=e_off * 0.55, fill_color=OFF, fill_opacity=0.85, stroke_width=0).move_to([-3.9, -2.0 + e_off * 0.275, 0])
        lb = VGroup(T(f"{v('err2d',1)}%", 20).next_to(b1, UP, 0.06), T(f"{v('err2d',0)}%", 20).next_to(b2, UP, 0.06),
                    T("dense mesh", 18, color=GREY_B).next_to(b1, DOWN, 0.1), T("off-mesh", 18, color=GREY_B).next_to(b2, DOWN, 0.1))
        tl = VGroup(T("worst error at 256²", 22, color=GREY_A), T("(accurate setting)", 18, color=GREY_B)).arrange(DOWN, buff=0.05).move_to([-4.35, 2.1, 0])
        self.play(FadeIn(tl), GrowFromEdge(b1, DOWN), GrowFromEdge(b2, DOWN), FadeIn(lb))
        self.say(f"On a coarse 256² mesh, the off-mesh solve's worst error is {v('err2d',0)}%, against {v('err2d',1)}% for the dense mesh solve, which carries the upwind error.", 0.5)

        inv = VGroup(T("per-case error change", 22, color=GREY_A), T("256² → 1024² → 4096²", 18, color=GREY_B),
                     T(f"≤ {v('spread2d')}", 40, color=OFF), T("(percentage points)", 16, color=GREY_B)).arrange(DOWN, buff=0.12).move_to([-1.3, 0.4, 0])
        self.play(FadeIn(inv, lag_ratio=0.25))
        self.say("Across meshes the error of each case barely moves: mesh-invariant, for both linear settings.")

        ax = Axes(x_range=[0, 2, 1], y_range=[0, 60, 20], x_length=3.6, y_length=3.2, tips=False,
                  axis_config={"color": GREY_C}).move_to([3.0, -0.2, 0])
        xl = VGroup(*[T(s, 18, color=GREY_B).next_to(ax.c2p(i, 0), DOWN, 0.12) for i, s in enumerate(["256²", "1024²", "4096²"])])
        yl = VGroup(*[T(f"{k}", 16, color=GREY_B).next_to(ax.c2p(0, k), LEFT, 0.1) for k in (0, 20, 40, 60)])
        at = T("solve time per query, ms", 20, color=GREY_A).next_to(ax, UP, 0.25)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(at))
        for key, col, name in [("t2d_acc", OFF, "Gauss 96²"), ("t2d_lat", MESH, "63² lattice"), ("t2d_fast", SOLVE, "Fib. 1597")]:
            vals = [float(x) for x in SRC_vals(key)]
            p = [ax.c2p(i, y) for i, y in enumerate(vals)]
            l = VMobject(color=col, stroke_width=4).set_points_as_corners(p)
            lab = T(name, 17, color=col).next_to(p[-1], RIGHT, 0.12)
            self.play(Create(l), FadeIn(lab), run_time=0.8)
        self.say(f"Solve time is flat from 256² to 4096²: across all 40 rules the ratio stays between {v('b4',0)} and {v('b4',1)}. Writing the answer back onto the mesh still grows with N.", 1)
        self.say(f"Not everything carried over: in the nonlinear 'head' setting a few cases move between meshes, and the chosen rule failed its test ({v('head2d',2)}% vs a 0.5% bar).", 1.5)
        self.clear_all()


# ================================================================ 8 · summary
class S8Summary(Base):
    def construct(self):
        self.header("Summary")
        items = [
            ("What Hari did", "Computed the advection integral with a fixed quadrature rule at points off the mesh, using the bank's exact derivatives.", OFF),
            ("Why it works", "The integrand is smooth: Gauss and lattice rules converge very fast. No fitting, no training data, nothing per-mesh.", BANK),
            ("The twist", "Our old rules copied the mesh's first-order upwind error; the off-mesh rule targets the true integral.", MESH),
            ("3D payoff", f"Error {v('err_off')}% at every mesh (tensor {v('err_tensor',0)}→{v('err_tensor',2)}%), faster queries, {v('mem',0)} → {v('mem',1)}; 256³ speedup {v('speedup',0)} → {v('speedup',1)} ({v('speedup',2)} at matched accuracy).", SOLVE),
            ("2D payoff", "Mesh-invariant error and flat solve time for the linear settings; the head setting did not carry over.", SOLVE),
            ("Still open", "Our bank needs many more points than Hari's model; first-order reference; head-setting drift; at 64³ still slower than the full solver.", GREY_B),
        ]
        rows = VGroup()
        for k, (h, body, col) in enumerate(items):
            hh = T(h, 26, color=col)
            bb = Paragraph(*wrap(body, 70), font=SANS, font_size=20, color=GREY_A, line_spacing=0.6)
            rows.add(VGroup(hh, bb).arrange(RIGHT, aligned_edge=UP, buff=0.4))
        for r in rows:
            r[0].set_width(r[0].width)
        hw = max(r[0].width for r in rows)
        for r in rows:
            r[1].next_to(r[0], RIGHT, buff=0.4 + hw - r[0].width, aligned_edge=UP)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32).next_to(self.hdr, DOWN, 0.5, aligned_edge=LEFT)
        if rows.height > 5.9:
            rows.scale_to_fit_height(5.9).next_to(self.hdr, DOWN, 0.4, aligned_edge=LEFT)
        for r in rows:
            self.play(FadeIn(r, shift=0.2 * RIGHT), run_time=0.7)
            self.wait(0.6 + len(r[1].submobjects) * 1.6)
        self.wait(3)
        src = T("Numbers: reports/2026-10-01-burgers3d-offmesh-quadrature.md and …-burgers2d-offmesh-quadrature.md", 16, color=GREY_C, font=SANS).to_edge(DOWN, 0.2)
        self.play(FadeIn(src))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


def wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines
