"""
    logmeanexp(A::AbstractArray; dims=:)

Computes `log.(mean(exp.(A); dims))`, in a numerically stable way.
"""
function logmeanexp(A::AbstractArray; dims=:)
    R = logsumexp(A; dims=dims)
    N = convert(eltype(R), length(A) ÷ length(R))
    return R .- log(N)
end

"""
    logvarexp(A::AbstractArray; dims=:)

Computes `log.(var(exp.(A); dims))`, in a numerically stable way.
"""
function logvarexp(
    A::AbstractArray; dims=:, corrected::Bool=true, logmean=logmeanexp(A; dims=dims)
)
    R = logsumexp(2logsubexp.(A, logmean); dims=dims)
    N = convert(eltype(R), length(A) ÷ length(R))
	if corrected
		return R .- log(N - 1)
    else
        return R .- log(N)
    end
end

"""
    logstdexp(A::AbstractArray; dims=:)

Computes `log.(std(exp.(A); dims))`, in a numerically stable way.
"""
function logstdexp(
    A::AbstractArray; dims=:, corrected::Bool=true, logmean=logmeanexp(A; dims=dims)
)
    return logvarexp(A; dims=dims, corrected=corrected, logmean=logmean) / 2
end
