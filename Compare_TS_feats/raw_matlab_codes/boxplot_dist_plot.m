close all
clear all
clc

cd('Z:\userdata\ekennedy\scripts\temp\heatervoltage')
load('chem_ts_feat_hv.mat')

num_sens = 8;
nf = 7642;
ylims = [.1 100];
plotting = 0;
sep_thresh = 0.5;

num_f = length(chem_ts_feat_hv.feat_mat.EtOH);
dist_maxmin = zeros(num_f,1);
sep = zeros(num_f,1);
feature_list = 1:num_f

for i = feature_list
    
    if rem(i,10000)==0
        disp([num2str(i),'/',num2str(num_f)])
    end
     
    feature_vals = [chem_ts_feat_hv.feat_mat.EtOH(i,:) ; ...
                    chem_ts_feat_hv.feat_mat.Ace(i,:)]';
    
    min_f = min(feature_vals);
    max_f = max(feature_vals);
    
    % if the lowest value in 1 is higher than all values in 2
    dist_sep(1) = (min_f(1) - max_f(2)); %>0
    % or if the lowest value in 2 is higher than all values in 1
    dist_sep(2) = (min_f(2) - max_f(1)); %>0
    % Then some threshold of this feature will perfectly seperate the 20x2
    % observations, so we can condense this to a binary sum check:
    sep(i) = sum(((min_f(1) - max_f(2)) > 0) + ((min_f(2) - max_f(1)) > 0));
    ms = median(feature_vals);
    ms = abs(ms(2) - ms(1));
    % With a distance metric 0.5x|max1 - min2|/|median1 - median2|
    dist_maxmin(i) = 0.5*max(dist_sep)/ms;
    chen_1_larger = (dist_maxmin>0);
    
    if plotting == 1
        
        if sep(i) == 1
            disp_sep = 'Yes';
        else
            disp_sep = 'No';
        end 
        boxplot(feature_vals,'Whisker',100)
        set(gcf,'color','white')
        axis square
        xlim([.5 2.5])
        title(['Feature #',num2str(i),'. Identifying: ',disp_sep,...
            '. Distance: ',num2str(100*dist_maxmin(i)),'%'])
        k = waitforbuttonpress;
        clf
        
    else
    end

end

figure
    plot(sep_inds,sep_feats,'.')
    set(gcf,'color','white')
    set(gca,'yscale','log')
    ylim(ylims)
    hold on
    steps = [num_f,num_f]/num_sens;
    for i = 1:num_sens  
        plot(i*steps,ylims,'k--')
        plot(i*steps - steps/2,ylims,'c--')  
    end
    xlim([0,num_f])
    ylabel('Distance, [max_c_1 - min_c_2] / |med_c_1_,_2|')
    xlabel('Feature #')
    
    
sep_feats = abs(dist_maxmin(sep==1));
sep_inds = find(sep==1);

use_f = find(sep_feats>sep_thresh);

subset_inds = sep_inds(use_f);
subset_feats = sep_feats(use_f);
figure
[y,x] = hist((rem(subset_inds,nf)),.5:.5:nf)

for i = 1:10
fs = randsample(sep_inds,2);
figure
    scatter(chem_ts_feat_hv.feat_mat.Ace(fs(1),:),chem_ts_feat_hv.feat_mat.Ace(fs(2),:),'fill','r')
    hold on
    scatter(chem_ts_feat_hv.feat_mat.EtOH(fs(1),:),chem_ts_feat_hv.feat_mat.EtOH(fs(2),:),'fill','bs')
    title('Acetone: red, EtOH: blue')
    xlabel(num2str(fs(1)))
    ylabel(num2str(fs(2)))
end

