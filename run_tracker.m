function [precision, overlap, fps, fn] = run_tracker(dataset, video, kernel_type, feature_type, show_visualization, show_plots,...
    padding,lambda, output_sigma_factor,interp_factor,kernel_sigma,cell_size,hog_orientations,mu,maxitr,mu_inc,step_sc)
	
	%path to the videos (you'll be able to choose one with the GUI).
	%base_path = strcat(pwd,'/');
    base_path = '/media/cjh/datasets/tracking/';
	result_path = 'results/';
	tracker_name = 'KCF_MTSA';
	thresholdPrecision = 20;
	thresholdOverlap = 0.5;

    %default settings
    if nargin < 1, dataset = 'OTB100'; end
    if nargin < 2, video = 'choose'; end
    if nargin < 3, kernel_type = 'gaussian'; end
    if nargin < 4, feature_type = 'hog'; end
    if nargin < 5, show_visualization = ~strcmp(video, 'all'); end
    if nargin < 6, show_plots = ~strcmp(video, 'all'); end
    
    if nargin < 7, padding = 1.7; end %extra area surrounding the target
    if nargin < 8, lambda = 1e-3; end %regularization
    if nargin < 9, output_sigma_factor = 0.1; end %spatial bandwidth (proportional to target)
    if nargin < 16, mu_inc = 2; end
    if nargin < 17, step_sc = 1e-2; end
    
    %parameters based on the chosen kernel or feature type
    kernel.type = kernel_type;
    features.gray = false;
    features.hog = false;
    features.hogcolor = false;
    %select feature type
	switch feature_type
    case 'gray',
        if nargin < 10, interp_factor = 0.075; end
        %interp_factor = 0.075;  %linear interpolation factor for adaptation
        if nargin < 11, kernel_sigma = 0.2; end %gaussian kernel bandwidth
        kernel.sigma = kernel_sigma;
        kernel.poly_a = 1;  %polynomial kernel additive term
        kernel.poly_b = 7;  %polynomial kernel exponent   
        if nargin < 12, cell_size = 1; end
        hog_orientations = 0;
        if nargin < 14, mu = 1e-6; end
        if nargin < 15, maxitr = 10; end      
        features.gray = true;
        
    case 'hog',
        if nargin < 10, interp_factor = 0.01; end        
        if nargin < 11, kernel_sigma = 0.5; end %gaussian kernel bandwidth
        kernel.sigma = kernel_sigma;        
        kernel.poly_a = 1;
        kernel.poly_b = 9;       
        if nargin < 12, cell_size = 4; end      
        if nargin < 13, hog_orientations = 9; end
        features.hog_orientations = hog_orientations;
        if nargin < 14, mu = 1e-6; end
        if nargin < 15, maxitr = 10; end    
        features.hog = true;
        
    case 'hogcolor',
        if nargin < 10, interp_factor = 0.01; end        
        if nargin < 11, kernel_sigma = 0.5; end %gaussian kernel bandwidth
        kernel.sigma = kernel_sigma;        
        kernel.poly_a = 1;
        kernel.poly_b = 9;        
        if nargin < 12, cell_size = 4; end       
        if nargin < 13, hog_orientations = 9; end
        features.hog_orientations = hog_orientations;
        if nargin < 14, mu = 1e-6; end
        if nargin < 15, maxitr = 10; end     
        features.hogcolor = true;       
    otherwise
        error('Unknown feature.')       
    end
	assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')

	switch video
    case 'choose',
        %ask the user for the video, then call self with that video name.
        video = choose_video([base_path dataset '/']);
        if ~isempty(video),
            [~, overlap, fps, fn] = run_tracker(dataset, video, kernel_type, feature_type, show_visualization, show_plots,...
                padding,lambda, output_sigma_factor,interp_factor,...
                kernel_sigma,cell_size,hog_orientations,mu,maxitr,mu_inc,step_sc);
        end 
        %we were given the name of a single video to process.
    otherwise
        %get image file names, initial state, and ground truth for evaluation
        [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, dataset,  video);
        [positions, rects, time] = tracker(video_path, img_files, pos, target_sz, ...
            padding, kernel, lambda, output_sigma_factor, interp_factor, ...
            cell_size, features, show_visualization,mu,maxitr,mu_inc,step_sc);
        
        %calculate precision, overlap and fps
        if (~exist ('rects','var'))
            target_sz_vec = repmat(fliplr(target_sz),[length(positions) 1]);
            rects = [fliplr(positions) - target_sz_vec/2, target_sz_vec];
        end
        
        [~, ~,errCoverage, errCenter] = calcSeqErrRobust(rects, ground_truth);
        overlap = sum(errCoverage > thresholdOverlap)/length(errCoverage);
        precision = sum(errCenter <= thresholdPrecision)/length(errCenter);
        fn = numel(img_files);
        fps = fn / time;
        fprintf('%12s - Precision(%upx): %.3f, Overlap(%u%%): %.3f, FPS: %.4g,\tFN: %u \n', video, thresholdPrecision, precision, thresholdOverlap*100, overlap, fps, fn)
    end
end