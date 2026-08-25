/**
 * Ink trail.
 *
 * A brush-like stroke that follows the pointer and fades out. It is decoration:
 * it does not communicate hierarchy, state or feedback, and the design guidance
 * this site otherwise follows discourages pointer-tracking effects. It ships
 * because it was asked for, with the cost side kept as close to zero as the
 * technique allows:
 *
 *   - The native cursor is untouched, so hit targets and affordances are intact.
 *   - The canvas is pointer-events:none and aria-hidden.
 *   - Nothing runs on touch or coarse-pointer devices; the script exits before
 *     creating any DOM.
 *   - prefers-reduced-motion: reduce disables it, live.
 *   - The rAF loop stops completely once the last point has decayed, so an idle
 *     tab costs nothing.
 *   - No scroll listener; the canvas is viewport-fixed and never reads layout.
 *
 * Tuning lives in CSS so it can be changed without touching this file:
 *   --ink-trail-alpha   peak opacity of the stroke      (default 0.14)
 *   --ink-trail-life    fade duration in ms             (default 600)
 *   --ink-trail-width   stroke width at zero speed, px  (default 3.6)
 */
(function () {
  'use strict';

  var fine = window.matchMedia('(hover: hover) and (pointer: fine)');
  if (!fine.matches) return;

  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');

  var canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  canvas.style.cssText =
    'position:fixed;inset:0;z-index:9989;pointer-events:none;';
  document.body.appendChild(canvas);
  var ctx = canvas.getContext('2d');

  var dpr = 1;
  var vw = 0;
  var vh = 0;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    vw = window.innerWidth;
    vh = window.innerHeight;
    canvas.width = Math.round(vw * dpr);
    canvas.height = Math.round(vh * dpr);
    canvas.style.width = vw + 'px';
    canvas.style.height = vh + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();

  var resizeTimer = 0;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 120);
  }, { passive: true });

  // ---- tunables, re-read whenever the theme flips ----
  var colour = '199,95,70';
  var alpha = 0.16;
  var life = 600;
  var baseWidth = 5;

  // How sharply the stroke thins with speed. Calibrated so an unhurried pointer
  // (~0.3 px/ms) keeps most of its width, a normal traverse (~1 px/ms) sits at
  // roughly 60%, and a fast flick (~4 px/ms) tapers to about a quarter. Higher
  // values were tried first and collapsed every ordinary movement to the
  // minimum width, which read as a hairline rather than a brush.
  var SPEED_K = 0.6;
  var MIN_WIDTH = 0.8;

  function num(styles, prop, fallback) {
    var v = parseFloat(styles.getPropertyValue(prop));
    return isNaN(v) ? fallback : v;
  }

  function readTheme() {
    var s = getComputedStyle(document.documentElement);
    alpha = num(s, '--ink-trail-alpha', 0.16);
    life = num(s, '--ink-trail-life', 600);
    baseWidth = num(s, '--ink-trail-width', 5);

    // --accent is a hex literal; resolve it to "r,g,b" via a throwaway element
    // so the canvas can vary only the alpha channel per segment.
    var probe = document.createElement('span');
    probe.style.cssText = 'position:absolute;visibility:hidden;color:var(--accent)';
    document.body.appendChild(probe);
    var m = getComputedStyle(probe).color.match(/(\d+)[,\s]+(\d+)[,\s]+(\d+)/);
    probe.remove();
    if (m) colour = m[1] + ',' + m[2] + ',' + m[3];
  }
  readTheme();

  new MutationObserver(readTheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class', 'data-theme'],
  });

  /* Tuning is cached at load, so changing the custom properties from devtools
     needs a nudge to take effect without a reload:
       document.documentElement.style.setProperty('--ink-trail-width', 7);
       inkTrail.refresh();                                                    */
  window.inkTrail = { refresh: readTheme };

  // ---- point buffer ----
  var MAX = 26;
  var pts = [];
  var running = false;
  var lastX = 0;
  var lastY = 0;
  var lastT = 0;

  function onMove(e) {
    if (reduce.matches) return;
    var now = performance.now();
    var dt = now - lastT;
    var w = baseWidth;

    if (lastT && dt > 0) {
      var dx = e.clientX - lastX;
      var dy = e.clientY - lastY;
      var speed = Math.sqrt(dx * dx + dy * dy) / dt; // px per ms
      // A dry brush thins as it moves faster; a slow one pools ink.
      w = baseWidth / (1 + speed * SPEED_K);
    }

    lastX = e.clientX;
    lastY = e.clientY;
    lastT = now;

    pts.push({ x: e.clientX, y: e.clientY, t: now, w: Math.max(MIN_WIDTH, Math.min(w, baseWidth)) });
    if (pts.length > MAX) pts.shift();

    if (!running) {
      running = true;
      requestAnimationFrame(draw);
    }
  }

  function draw(now) {
    ctx.clearRect(0, 0, vw, vh);

    // drop decayed points
    while (pts.length && now - pts[0].t > life) pts.shift();

    if (pts.length < 2 || reduce.matches) {
      if (!pts.length || reduce.matches) {
        ctx.clearRect(0, 0, vw, vh);
        running = false; // idle: stop scheduling frames entirely
        return;
      }
      requestAnimationFrame(draw);
      return;
    }

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    /* Stroked in a handful of runs rather than segment by segment.
       One stroke() per segment composites each round cap over its neighbour,
       which doubles the alpha at every join and turns the stroke into a string
       of beads at anything above a whisper. A single stroke() composites its
       whole path once, so grouping consecutive points into a few age buckets
       removes the beading while keeping the fade and the taper. */
    var n = pts.length;
    var BUCKETS = 6;
    var per = Math.max(1, Math.ceil((n - 1) / BUCKETS));

    for (var s = 0; s < n - 1; s += per) {
      var e = Math.min(n - 1, s + per);
      var midIdx = (s + e) / 2;
      var mid = pts[Math.floor(midIdx)];
      var age = (now - mid.t) / life;
      if (age >= 1) continue;

      // taper: older runs fade, and the tail is thinner than the head
      var head = midIdx / (n - 1);
      var a = alpha * (1 - age) * (0.35 + 0.65 * head);

      ctx.beginPath();
      ctx.moveTo(pts[s].x, pts[s].y);
      for (var i = s + 1; i <= e; i++) {
        var p = pts[i];
        var q = pts[i - 1];
        ctx.quadraticCurveTo(q.x, q.y, (q.x + p.x) / 2, (q.y + p.y) / 2);
      }
      ctx.lineTo(pts[e].x, pts[e].y);
      ctx.strokeStyle = 'rgba(' + colour + ',' + a.toFixed(4) + ')';
      ctx.lineWidth = mid.w * (0.4 + 0.6 * head);
      ctx.stroke();
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener('pointermove', onMove, { passive: true });

  // Clear immediately if the user turns reduced motion on mid-session.
  reduce.addEventListener('change', function () {
    if (reduce.matches) {
      pts.length = 0;
      ctx.clearRect(0, 0, vw, vh);
    }
  });

  // A pointer that leaves the window should not leave a frozen stroke behind.
  window.addEventListener('blur', function () {
    pts.length = 0;
  });
})();
