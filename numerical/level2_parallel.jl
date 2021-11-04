using QuadGK
using LambertW
using Base.Threads

"""
Compute range y1, y2 of the Converse Jensen Inequality:
from 0 to y1: ρ1 sum to x1;
from 0 to y2: ρ2 sum to x2.
"""
function converse_jensen_range(x1::Float64, x2::Float64, xmax::Float64=1-1e-10)::Tuple{Float64,Float64}
    # Cap x1, x2 by xmax because LambertW package may have large error near 1/ℯ
    x1 = min(x1, xmax)
    x2 = min(x2, xmax)
    if 1-(1-x1)*(1-log(1-x1)) >= x2
        # case 1: λ2 <= λ1.
        # Solving 1-(1+λ)exp(-λ)=x2 gives λ2=-1-W(-(1-x2)/ℯ) (-1 branch)
        λ2 = -1-lambertw(-(1-x2)/ℯ, -1)
    else
        # case 1: λ2 > λ1
        # Solving 2-(2+λ)exp(-λ)=x1+x2 gives λ2=-2-W(-(2-x1-x2)/ℯ^2) (-1 branch)
        λ2 = -2-lambertw(-(2-x1-x2)/ℯ^2, -1)
    end
    y1 = x1
    y2 = min(1-exp(-λ2), xmax)
    return y1, y2
end

"""Compute the contribution at any y = 1-exp(-λ) to (d/dt) log(d_(x1, x2)(t))."""
function dlog_double_dy(y::Float64, y1::Float64, y2::Float64, x1::Float64, x2::Float64, t::Float64, logd::Float64)::Float64
    ρ1 = (y<=y1) ? 1.            : 0.
    ρ2 = (y<=y2) ? 1-log(1-y)-ρ1 : 0.
    w1 = exp((2*x1+x2)*t+logd)
    w2 = exp((x1+2*x2)*t+logd)
    return (w1*ρ1+w2*ρ2)/((max(w1-1,0.)*ρ1+max(w2-1,0.)*ρ2)*(1-y)+1)
end

"""Compute the value of log(q_x(t)) for all 0 <= t <=1 that is a multiple of dt, for a given x."""
function logq(x::Float64, dx::Float64=1e-3, dt::Float64=1e-3, xmax::Float64=1-1e-10, rtol::Float64=1e-5)::Array{Float64,1}

    x_grid = Array(0.:dx:1-dx)
    nx = length(x_grid)

    t_grid = Array(0.:dt:1-dt)
    nt = length(t_grid)

    logq_grid = zeros(nx,nt)

    @threads for i in 1:nx
        x1, x2 = (x>=x_grid[i]) ? [x, x_grid[i]] : [x_grid[i], x]
        y1, y2 = converse_jensen_range(x1, x2, xmax)
        logd = 0.
        for j in 1:(nt-1)
            t = t_grid[j]
            dlogd, err = quadgk(y -> dlog_double_dy(y, y1, y2, x1, x2, t, logd), 0., xmax, rtol=rtol)
            logd2      = logd-(dlogd-err)*dt
            dlogd, err = quadgk(y -> dlog_double_dy(y, y1, y2, x1, x2, t, logd2), 0., xmax, rtol=rtol)
            logd      -= (dlogd-err)*dt
            logd       = min(logd, -(x1+x2)*(t+dt))   # use trivial bound if numerical bound is worse (for small t)
            logq_grid[i,j+1] = min(logd+(x_grid[i]+dx)*(t+dt), -x*(t+dt))
        end
    end

    return vec(maximum(logq_grid, dims=1))
end

"""Compute match probability bound of an vertex with LP matched level x"""
function level2(x::Float64=1., dx::Float64=1e-3, dt::Float64=1e-3, xmax::Float64=1-1e-10, rtol::Float64=1e-5)::Float64
    log_single = 0.
    t_grid = Array(0.:dt:1-dt)
    nt = length(t_grid)
    logq_grid = logq(x, dx, dt, xmax, rtol)
    for i in 1:nt
        t = t_grid[i]
        y = 1-exp(-x*t+logq_grid[i]-log_single)
        dlog_single = (y==0.) ? x : -log(1-x*y)/y
        log_single2 = log_single - dlog_single*dt
        y = 1-exp(-x*t+logq_grid[i]-log_single2)
        dlog_single = (y==0.) ? x : -log(1-x*y)/y        
        log_single -= dlog_single*dt
    end
    return 1-exp(log_single)
end
