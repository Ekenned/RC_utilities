close all
clear all
clc

cd('Z:\userdata\pratistha\TB\out_files\20190321_EtOH_Acetone')
load('TB_RComp_1.mat')

cd('Z:\userdata\ekennedy\scripts\Tools\hctsa-master')

dataset = 2;

MOX = TB_RComp(dataset).MOX;
P = TB_RComp(dataset).P;
T = TB_RComp(dataset).T;
VOC = 0.5 - MOX;
SR = TB_RComp(dataset).SamplingRate;

%MOX = MOX(:,TB_RComp(dataset).sort_order,:);
%VOC = VOC(:,TB_RComp(dataset).sort_order,:);
%P = P(:,TB_RComp(dataset).sort_order,:);
%T = T(:,TB_RComp(dataset).sort_order,:);

n = size(MOX);

%VOC = Truffle_RComp_EtOH_20190318.VOC;
%P = Truffle_RComp_EtOH_20190318.P;
%T = Truffle_RComp_EtOH_20190318.T;
labels = TB_RComp(dataset).pattern_labels;

num_traces = n(2);
num_sensors = n(1);
num_sensors = 8;

smoothing = 3; % pixels
sparsity = 1; % pixels

for sens = 1:num_sensors
    
    for trace_num = 1:num_traces
        
        % Isolate VOC,P,T for 1 sensor for 1 trace
        locl_vec_VOC = squeeze( VOC(sens,trace_num,:) );
        locl_vec_VOC  = smooth(locl_vec_VOC,smoothing);
        locl_vec_VOC = locl_vec_VOC(1:sparsity:end);
        locl_vec_P = squeeze(P(sens,trace_num,:) )-movmean(squeeze(P(sens,trace_num,:) ),SR*30);
        locl_vec_P  = smooth(locl_vec_P,smoothing);
        locl_vec_P = locl_vec_P(1:sparsity:end);
        locl_vec_T = squeeze(T(sens,trace_num,:) )-movmean(squeeze(T(sens,trace_num,:) ),SR*30);
        locl_vec_T  = smooth(locl_vec_T,smoothing);
        locl_vec_T = locl_vec_T(1:sparsity:end);
        
        % Perform feature extraction on V,P,T sepeartely
        feat_VOC{sens,trace_num} = TS_CalculateFeatureVector(locl_vec_VOC);               
        feat_P{sens,trace_num} = TS_CalculateFeatureVector(locl_vec_P);     
        feat_T{sens,trace_num} = TS_CalculateFeatureVector(locl_vec_T);  
        feat_VOC_d{sens,trace_num} = TS_CalculateFeatureVector(diff(smooth(locl_vec_VOC)));               
        feat_P_d{sens,trace_num} = TS_CalculateFeatureVector(diff(smooth(locl_vec_P),SR));     
        feat_T_d{sens,trace_num} = TS_CalculateFeatureVector(diff(smooth(locl_vec_T),SR));  
               
    end
    
    cd('Z:\userdata\ekennedy\scripts\temp')
    feature_struct2.feat_VOC =   feat_VOC;
    feature_struct2.feat_P =     feat_P;
    feature_struct2.feat_T =     feat_T;
    feature_struct2.feat_VOC_d =   feat_VOC_d;
    feature_struct2.feat_P_d =     feat_P_d;
    feature_struct2.feat_T_d =     feat_T_d;
    feature_struct2.labels =     labels;

    save feature_struct2 feature_struct2

end

% Build a matrix where every row is [V(s=1),P(s=1),T(s=1),V(s=2),P(s=2),T(s=2)...T(s=N)]
feat_mat_concat = []; % concatenated matrix of these rows where num_traces = columns
for trace_num = 1:num_traces
    
    feat_sens_concat = []; % concatenated vector of all sensors
    for sens = 1:num_sensors    
        % Get the combined V,P,T feature vector
        feat_VPT = [feat_VOC{sens,trace_num} ; feat_P{sens,trace_num} ; feat_T{sens,trace_num} ;...
                    feat_VOC_d{sens,trace_num} ; feat_P_d{sens,trace_num} ; feat_T_d{sens,trace_num}]; 
        feat_sens_concat = [feat_sens_concat ; feat_VPT];      
    end
    
    feat_mat_concat(:,trace_num) = feat_sens_concat;

end

num_feats = length(feat_mat_concat(:,1));

% normalize every feature list by the observed feature max and min
for feats = 1:num_feats
    
    lv = feat_mat_concat(feats,:);
    lv_max(feats) = max(feat_mat_concat(feats,:));
    lv_min(feats) = min(feat_mat_concat(feats,:));
    lv_norm(feats,:) = (lv - lv_min(feats))/(lv_max(feats) - lv_min(feats));
    
end
lv_norm(isnan(lv_norm))=0;
lv_norm = lv_norm';

cd('Z:\userdata\ekennedy\scripts\temp')

feature_struct2.dataset = dataset;
% feature_struct2.name = name;
feature_struct2.feat_mat_concat = feat_mat_concat;
feature_struct2.lv_norm =    lv_norm;
feature_struct2.feat_VOC =   feat_VOC;
feature_struct2.feat_P =     feat_P;
feature_struct2.feat_T =     feat_T;
feature_struct2.feat_VOC_d =   feat_VOC_d;
feature_struct2.feat_P_d =     feat_P_d;
feature_struct2.feat_T_d =     feat_T_d;
feature_struct2.labels =     labels;

save feature_struct2 feature_struct2






