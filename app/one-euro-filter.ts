import { Quaternion } from 'three';

const smoothingFactor = (t_e: number, cutoff: number) => {
  const r = 2 * Math.PI * cutoff * t_e;
  return r / (r + 1);
};

const exponentialSmoothing = (a: number, x: number, x_prev: number) => {
  return a * x + (1 - a) * x_prev;
};

export class OneEuroFilter {
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;

  private x_prev: number | null = null;
  private dx_prev: number = 0;
  private t_prev: number | null = null;

  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  reset() {
    this.x_prev = null;
    this.dx_prev = 0;
    this.t_prev = null;
  }

  filter(x: number, t: number): number {
    if (this.t_prev === null || this.x_prev === null) {
      this.x_prev = x;
      this.dx_prev = 0;
      this.t_prev = t;
      return x;
    }

    const t_e = (t - this.t_prev) / 1000.0;
    if (t_e <= 0) return this.x_prev;

    const a_d = smoothingFactor(t_e, this.dCutoff);
    const dx = (x - this.x_prev) / t_e;
    const dx_hat = exponentialSmoothing(a_d, dx, this.dx_prev);

    const cutoff = this.minCutoff + this.beta * Math.abs(dx_hat);
    const a = smoothingFactor(t_e, cutoff);
    const x_hat = exponentialSmoothing(a, x, this.x_prev);

    this.x_prev = x_hat;
    this.dx_prev = dx_hat;
    this.t_prev = t;

    return x_hat;
  }
}

export class Vector2OneEuroFilter {
  private xFilter: OneEuroFilter;
  private yFilter: OneEuroFilter;

  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.xFilter = new OneEuroFilter(minCutoff, beta, dCutoff);
    this.yFilter = new OneEuroFilter(minCutoff, beta, dCutoff);
  }

  reset() {
    this.xFilter.reset();
    this.yFilter.reset();
  }

  filter(point: { x: number; y: number }, t: number): { x: number; y: number } {
    return {
      x: this.xFilter.filter(point.x, t),
      y: this.yFilter.filter(point.y, t),
    };
  }
}

export class QuaternionOneEuroFilter {
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;

  private q_prev: Quaternion | null = null;
  private dq_prev: number = 0; // Magnitude of angular velocity
  private t_prev: number | null = null;

  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  reset() {
    this.q_prev = null;
    this.dq_prev = 0;
    this.t_prev = null;
  }

  filter(q: Quaternion, t: number): Quaternion {
    if (this.t_prev === null || this.q_prev === null) {
      this.q_prev = q.clone();
      this.dq_prev = 0;
      this.t_prev = t;
      return this.q_prev.clone();
    }

    const t_e = (t - this.t_prev) / 1000.0;
    if (t_e <= 0) return this.q_prev.clone();

    // Calculate angular distance between quaternions
    const dot = this.q_prev.dot(q);
    const absDot = Math.min(Math.abs(dot), 1.0);
    const angle = 2 * Math.acos(absDot);
    
    // Calculate angular velocity
    const dq = angle / t_e;
    
    // Filter angular velocity
    const a_d = smoothingFactor(t_e, this.dCutoff);
    const dq_hat = exponentialSmoothing(a_d, dq, this.dq_prev);
    
    // Calculate adaptive cutoff
    const cutoff = this.minCutoff + this.beta * Math.abs(dq_hat);
    const a = smoothingFactor(t_e, cutoff);
    
    // Slerp to filtered orientation
    // Make sure we take the shortest path
    const qCorrected = q.clone();
    if (dot < 0) {
      qCorrected.set(-q.x, -q.y, -q.z, -q.w);
    }
    
    const q_hat = this.q_prev.clone().slerp(qCorrected, a);

    this.q_prev = q_hat;
    this.dq_prev = dq_hat;
    this.t_prev = t;

    return q_hat.clone();
  }
}
