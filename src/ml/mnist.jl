using Flux: onehotbatch
using Flux.Data: DataLoader, MNIST

"""
    prepare_data(x, y, [isflat=false])
"""
function prepare_data(x, y; isflat=false)
    xx = Array{Float32}(undef, size(x[1])..., 1, length(x))
    for i in 1:length(x)
        xx[:, :, :, i] = Float32.(x[i])
    end
    
    xx = isflat ? Flux.flatten(xx) : xx
    y = onehotbatch(y, 0:9)
    return DataLoader(xx, y, batchsize=10, shuffle=true, partial=true)
end


"""
    get_traindata(indices, [isflat=false])

Prepare the MNIST training data for the specified range.
"""
function get_traindata(indices::UnitRange; isflat=false)
    xtrain = Flux.Data.MNIST.images()[indices]
    ytrain = Flux.Data.MNIST.labels()[indices]
    return prepare_data(xtrain, ytrain, isflat=isflat)
end


"""
    get_testdata([isflat=false])

Prepare the MNIST testing data for the specified range.
"""
function get_testdata(; isflat=false)
    xtest = Flux.Data.MNIST.images(:test)
    ytest = Flux.Data.MNIST.labels(:test)
    return prepare_data(xtest, ytest, isflat=isflat)
end

"""
    getdata(isflat=false)

Prepare the MNIST dataset.
"""
function getdata(isflat=false)
    traindata = get_traindata(1:60000, isflat=isflat)
    testdata = get_testdata(isflat=isflat)
    return traindata, testdata
end