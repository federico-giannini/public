#=
Subsets.jl

Author:         Federico Giannini
Date:           26-11-2025
Description:    This file handles the behaviour of the iterable "Subsets".
=#

struct Subsets{T}
    set::Vector{T}
    size::Int
end

function Base.iterate(iterable::Subsets)
    indices = [length(iterable.set) - i + 1 for i in 1 : iterable.size]
    subset = iterable.set[indices]
    return (subset, indices)
end

function Base.iterate(iterable::Subsets, indices::Vector{Int})
    k = iterable.size
    for i in k : -1 : 1
        if indices[i] > k - i + 1
            indices[i] -= 1
            for j = i + 1 : k
                indices[j] = indices[j - 1] - 1
            end
            subset = iterable.set[indices]
            return (subset, indices)
        end
    end
    return nothing
end

for subset in Subsets([0, 2, 4, 6], 4)
    println(subset)
end
