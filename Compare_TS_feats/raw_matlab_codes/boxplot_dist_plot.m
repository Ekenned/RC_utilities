close all
clear all
clc

cd('Z:\userdata\ekennedy\Plume\twuffle\RC\hv')
load('chem_ts_feat_hv.mat')

num_sens = 8;
nf = 7642;
ylims = [.1 100];
plotting = 0;
sep_thresh = 0.5;

num_f = length(chem_ts_feat_hv.feat_mat.EtOH);
dist_maxmin = zeros(num_f,1);
sep = zeros(num_f,1);
feature_list = 1:num_f;

for i = 1:length(feature_list)
    
    if rem(i,10000)==0
        disp([num2str(i),'/',num2str(num_f)])
    end
    
    % Make a 2 x num_obs matrix for the feature
    feature_vals = [chem_ts_feat_hv.feat_mat.EtOH(feature_list(i),:) ; ...
                    chem_ts_feat_hv.feat_mat.Ace(feature_list(i),:)]';
    
    % argmin( V_EtOH - V_Ace.T ), if feature two is bigger then
    if (median(feature_vals(:,2)) - median(feature_vals(:,1)))>0
        min_sep_dist = min(min(feature_vals(:,2) - feature_vals(:,1)'));
    else
        min_sep_dist = min(min(feature_vals(:,1) - feature_vals(:,2)'));
    
    end
    % Whether or not the data is perfectly seperating
    sep(i) = (min_sep_dist > 0);
    % halfway point between the two chemical medians
    ms = 0.5*abs(sum(median(feature_vals)));
    % The feature's 90/10 qunatile range:
    % ms = quantile(feature_vals(:),.9) - quantile(feature_vals(:),.1);
    % The range of the feature observed over all chemicals % ms = max_sep_dist;
    % Establish a distance metric, e.g. 0.5x|min_dist|/|median|
    dist_maxmin(i) = min_sep_dist/ms;
    
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
        title(['Feature #',num2str(i),'. Distance: ',num2str(100*dist_maxmin(i)),'%'])
        k = waitforbuttonpress;
        clf
        
    else
    end

end

sep_feats = abs(dist_maxmin(sep==1));
sep_inds = find(sep==1);
feature_list = sep_inds(find(sep_feats>.8));

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
    

use_f = find(sep_feats>=sep_thresh);

subset_inds = sep_inds(use_f);
subset_feats = sep_feats(use_f);
inf_rem = sep_feats(use_f) ~= inf;
subset_inds = subset_inds(inf_rem);
subset_feats = subset_feats(inf_rem);

disp(length(subset_inds))

figure
[y,x] = hist((rem(subset_inds,nf)),.5:.5:nf);
stem(x,y)
[a,b] = sort(y);
ranked_features = fliplr(b);
x = x(ranked_features);
figure
c = 0;
for i = 1:2:36%length(subset_inds)

    % fs = randsample(subset_inds,2);
    fs = subset_inds([i,i+1]);
    c = c+1;

    subplot(3,6,c)
    scatter(chem_ts_feat_hv.feat_mat.Ace(fs(1),:),chem_ts_feat_hv.feat_mat.Ace(fs(2),:),'fill','b')
    hold on
    scatter(chem_ts_feat_hv.feat_mat.EtOH(fs(1),:),chem_ts_feat_hv.feat_mat.EtOH(fs(2),:),'fill','rs')
    % title('Acetone: blue, EtOH: red')
    xlabel(num2str(fs(1)))
    ylabel(num2str(fs(2)))
    %lgd = legend([num2str(100*subset_feats(fs(1))),'%'],...
    %                [num2str(100*subset_feats(fs(2))),'%']);
    %lgd.FontSize = 9;
    axis square
    box on
    set(gcf,'color','white')

end

figure
    plot(fliplr(sort(subset_feats)'),'o')
    set(gcf,'color','white')
    set(gca,'yscale','log')
    set(gca,'xscale','log')
    ylabel('Feature minimum seperation distance')
    ylim([sep_thresh 100])
    axis square
    grid on

%sub_inds = subset_inds;
% subset_inds = subset_inds(find(subset_feats==max(subset_feats)));
figure
Ace_mat = chem_ts_feat_hv.feat_mat.Ace(subset_inds,:);
EtOH_mat = chem_ts_feat_hv.feat_mat.EtOH(subset_inds,:);
conc_mat = [Ace_mat , EtOH_mat ];
conc_mat(isnan(conc_mat)) = 0;
[coeff,score,latent] = pca(conc_mat);
plot3(coeff(21:40,1),coeff(21:40,2),coeff(21:40,3),'rs')
hold on
plot3(coeff(1:20,1),coeff(1:20,2),coeff(1:20,3),'bo') % acetone is blue circles
hold off
box on
grid on
set(gcf,'color','white')

feat_analysis.sub_inds = subset_inds;
feat_analysis.sub_powers = subset_feats;

feat_analysis

save feat_analysis feat_analysis
