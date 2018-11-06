################################################################################
## This script is for creating plots for the BNs in RC project
## Current plots include:
##
################################################################################
using Plots

filepath = "C:/Users/ChuperDuper/Documents/Thesis/Experiments/Simulations/ReservoirComputers"
N = [10; 20; 30; 40; 50; 100; 200; 300; 400; 500]
L = [10:10:100;]
K = 2
numN = length(N)
numL = length(L)
numRes = 100
numF = 2^2^3

# Import Data
data = readcsv("$(filepath)/output_all3bit_bias0.5_training150_stream10_delay1_$(numRes)experiments.csv")
data_f64 = convert(Array{Float64,2},data[2:end,:])

# Mean accuracy for median function
# Median function ID = 23
funID = 23
#takemeanoverR(permutetoNxLxR(reshapeintoLxNxR(pull out data only for median)))
data_metric = mean(permutedims(reshape(data_f64[data_f64[:,5].==funID,6],numL,numN,numRes),[2,1,3]),3)
fig = heatmap(reshape(data_metric,numN,numL),
    title = "Mean Accuracy - 3-bit Median",
    clim = (0.5,1),
    #color=:gray,
    colorbar_title="Mean Accuracy",
    xtickfont = font(10,"TimesNewRoman"),
    xlabel = "L - % Nodes Connected to Input Layer",
    xtick =  ([1:numL;],string.(unique(data[2:end,4]))),
    ytickfont = font(10,"TimesNewRoman"),
    ytick =  ([1:numN;],string.(unique(data[2:end,3]))),
    ylabel="Size of Reservoir",
    guidefont = font(12))

savefig(fig,"$(filepath)/K$(K)_MeanAccuracy_median")

# Mean accuracy for parity function
# Parity function ID = 105
funID = 105
#takemeanoverR(permutetoNxLxR(reshapeintoLxNxR(pull out data only for median)))
data_metric = mean(permutedims(reshape(data_f64[data_f64[:,5].==funID,6],length(L),length(N),numRes),[2,1,3]),3)
fig = heatmap(reshape(data_metric,numN,numL),
    title = "Mean Accuracy - 3-bit Parity",
    clim = (0.5,1),
    #color=:gray,
    colorbar_title="Mean Accuracy",
    xtickfont = font(10,"TimesNewRoman"),
    xlabel = "L - % Nodes Connected to Input Layer",
    xtick =  ([1:numN;],string.(unique(data[2:end,4]))),
    ytickfont = font(10,"TimesNewRoman"),
    ytick =  ([1:numL;],string.(unique(data[2:end,3]))),
    ylabel="Size of Reservoir",
    guidefont = font(12))

savefig(fig,"$(filepath)/K$(K)_MeanAccuracy_parity")

# Mean accuracy for all functions
#takemeanoverRandF(permutetoNxLxFxR(reshapeintoFxLxNxR))
data_metric = mean(permutedims(reshape(data_f64[:,6],numF,numL,numN,numRes),[3,2,1,4]),[3 4])
fig = heatmap(reshape(data_metric,numN,numL),
    title = "Mean Accuracy - All 3-bit Functions",
    clim = (0.5,1),
    #color=:gray,
    colorbar_title="Mean Accuracy",
    xtickfont = font(10,"TimesNewRoman"),
    xlabel = "L - % Nodes Connected to Input Layer",
    xtick =  ([1:numL;],string.(unique(data[2:end,4]))),
    ytickfont = font(10,"TimesNewRoman"),
    ytick =  ([1:numN;],string.(unique(data[2:end,3]))),
    ylabel="Size of Reservoir",
    guidefont = font(12))

savefig(fig,"$(filepath)/K$(K)_MeanAccuracy_all")


# Mean accuracy across all functions (with a given N,L)
#takemeanoverR(permutetoNxLxFxR(reshapeintoFxLxNxR))
data_metric = mean(permutedims(reshape(data_f64[:,6],numF,numL,numN,numRes),[3,2,1,4]),4)
fig = plot(layout=(numN,numL))
count = 0
for i = 1:numN  # N
    for j = 1:numL # L
        count+=1
        histogram!(data_metric[end-i+1,j,:,1], # go from end of rows because N goes increases from top to bottom
        subplot=count,
        #title = "",
        #nbins = 11, #A
        nbins = [0:0.1:1.1;], #B
        #xlim = (-0.05,1.15),
        ylim = (0,numF),
        legend = false,
        #ylabel = "# of Tissues",
        #xlabel = "log # of All Observed CAs",
        #ytickfont = font(30,"TimesNewRoman"),
        ytick = false,
        #xtickfont = font(30,"TimesNewRoman"),
        #guidefont = font(30),
        #titlefont = font(30),
        xtick = false)
    end
end
plot(fig)
savefig(fig,"$(filepath)/K$(K)_MeanAccuracy_functionHistA")


# Histogram of accuracies for a single reservoirs
data_metric = permutedims(reshape(data_f64[:,6],numF,numL,numN,numRes),[3,2,1,4])
for i = 1:numN # N
    for j = 1:numL # L
        fig = plot(layout=(10,10))
        for k = 1:numRes  # Reservoir ID
            histogram!(data_metric[i,j,:,k],
                subplot=k,
                #title = "",
                nbins = [0:0.1:1.1;],
                xlim = (-0.05,1.15),
                ylim = (0,numF),
                legend = false,
                #ylabel = "# of Tissues",
                #xlabel = "log # of All Observed CAs",
                #ytickfont = font(30,"TimesNewRoman"),
                ytick = false,
                #xtickfont = font(30,"TimesNewRoman"),
                #guidefont = font(30),
                #titlefont = font(30),
                xtick = false)
        end
        savefig(fig,"$(filepath)/MeanAccuracy_functionHist_N$(N[i])L$(L[j])")
    end
end
# histogram for 1 network
fig = histogram(data_metric[10,10,:,1],
    title = "N = $(N[10]), L = $(L[10])",
    nbins = [0:0.1:1.1;],
    xlim = (-0.0,1.15),
    ylim = (0,numF),
    legend = false,
    ylabel = "# of Functions",
    xlabel = "Accuracy",
    ytickfont = font(20,"TimesNewRoman"),
    ytick = true,
    xtickfont = font(20,"TimesNewRoman"),
    guidefont = font(20),
    titlefont = font(20),
    xtick = true)
    savefig(fig,"$(filepath)/MeanAccuracy_functionHist_N$(N[10])L$(L[10])R1")

# Histogram of sum(# of >90% accuracy functions) across all reservoirs
data_metric = sum(permutedims(reshape(data_f64[:,6],numF,numL,numN,numRes),[3,2,1,4]).>0.9,3)./numF
fig = plot(layout=(numN,numL))
count = 0
for i = 1:numN  # N
    for j = 1:numL # L
        count+=1
        histogram!(data_metric[end-i+1,j,1,:],
        subplot=count,
        #title = "",
        nbins = [0:0.1:1.1;],
        xlim = (-0.05,1.15),
        ylim = (0,numRes),
        legend = false,
        #ylabel = "# of Tissues",
        #xlabel = "log # of All Observed CAs",
        #ytickfont = font(30,"TimesNewRoman"),
        ytick = false,
        #xtickfont = font(30,"TimesNewRoman"),
        #guidefont = font(30),
        #titlefont = font(30),
        xtick = false)
    end
end
savefig(fig,"$(filepath)/K$(K)_MeanAccuracy_GT90AccuracyHist")

# Mean accuracy Vs. Function Sensitivity
f_sens = readdlm("$(filepath)/AverageSensitivities_all3bit.txt",Float64)
data_metric = mean(permutedims(reshape(data_f64[:,6],numF,numL,numN,numRes),[3,2,1,4]),4)
# For NxL grid of subplots
fig = plot(layout=(numN,numL))
count = 0
for i = 1:numN  # N
    for j = 1:numL # L
        count+=1
        scatter!(f_sens[:],data_metric[end-i+1,j,:,1],
        subplot=count,
        #title = "",
        xlim = (-0.2,3.2),
        #ylim = (-0.1,1.1), #A
        ylim = (minimum(data_metric[end-i+1,j,:,1]),maximum(data_metric[end-i+1,j,:,1])), #B
        legend = false,
        #ylabel = "# of Tissues",
        #xlabel = "log # of All Observed CAs",
        #ytickfont = font(30,"TimesNewRoman"),
        ytick = false,
        #xtickfont = font(30,"TimesNewRoman"),
        #guidefont = font(30),
        #titlefont = font(30),
        xtick = false)
    end
end
savefig(fig,"$(filepath)/K$(K)_SensitivityVsMeanAccuracyB")

# For 3 L values on one plot
for i = [1 5 10]  # N
    fig = plot()
    for j = [1 5 10] # L
        scatter!(f_sens[:],data_metric[i,j,:,1],
        markersize = 12,
        title = "N = $(N[i])",
        xlim = (-0.2,3.2),
        ylim = (-0.1,1.1),
        legend = :bottomleft,
        legendfont = font(20,"TimesNewRoman"),
        label = ("L = $(L[j])"),
        xlabel = "Function Average Sensitivty",
        ylabel = "Mean Accuracy",
        ytickfont = font(20,"TimesNewRoman"),
        ytick = true,
        xtickfont = font(20,"TimesNewRoman"),
        guidefont = font(20),
        titlefont = font(20),
        xtick = true)
    end
    savefig(fig,"$(filepath)/K$(K)_SensitivityVsMeanAccuracy_N$(N[i])")
end



# mean accuracy v function ID (should be symmetrical)
data_metric = mean(permutedims(reshape(data_f64[:,6],numF,numL,numN,numRes),[3,2,1,4]),4)
fig = plot(layout=(numN,numL))
count = 0
for i = 1:numN  # N
    for j = 1:numL # L
        count+=1
        bar!(data_metric[end-i+1,j,:,1],
        subplot=count,
        #title = "",
        #xlim = (-0.2,3.2),
        #ylim = (-0.1,1.1),
        #ylim = (minimum(data_metric[end-i+1,j,:,1]),maximum(data_metric[end-i+1,j,:,1])),
        legend = false,
        #ylabel = "# of Tissues",
        #xlabel = "log # of All Observed CAs",
        #ytickfont = font(30,"TimesNewRoman"),
        ytick = false,
        #xtickfont = font(30,"TimesNewRoman"),
        #guidefont = font(30),
        #titlefont = font(30),
        xtick = false)
    end
end
plot(fig)
