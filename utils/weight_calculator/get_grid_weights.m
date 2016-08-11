function [v, w] = get_grid_weights(R, Roff, dir, stamp, bank, idx, overwrite)

    weight_mat = sprintf('%s/%s%s_weights.mat', dir, stamp, bank);
    disp('Computing normalized steering vectors and beamformer weights');
    Nbin = size(R, 3);
    Ntime = size(R, 4);
    
    if ~exist(weight_mat, 'file') || overwrite
        v = zeros(size(Roff,1), Nbin, Ntime);
        w = zeros(size(Roff,1), Nbin, Ntime);
        for t = 1:Ntime
            if mod(t, 100) == 0
                fprintf('\tTime %d\n', t);
            end
            v(idx,:,t) = gen_steering_vectors(squeeze(R(idx,idx,:,t)), Roff(idx,idx,:));
            w(idx,:,t) = get_maxSNR_weights(v(idx,:,t), Roff(idx,idx,:));
        end
        save(weight_mat, 'v', 'w');
    else
        load(weight_mat);
    end

end

function [ v ] = gen_steering_vectors( Ron, Roff )
%GEN_STEERING_VECTORS Estimates array steering vectors based on
%on/off-source array covariance matrices
%   Detailed explanation goes here

    % Make sure the matrices are the same size
    if prod(size(Ron) == size(Roff)) == 0 
        error(sprintf(['Covariance matrices must be the same size!\n',...
               '\t size(Ron)  = [', num2str(size(Ron)),  ']\n',...
               '\t size(Roff) = [', num2str(size(Roff)), ']']));
    end
    
    % Make sure that the matrices are square
    if size(Ron, 1) ~= size(Ron, 2)
        error(sprintf(['Covariance matrices must be square!\n',...
               '\t size(Ron) = [', num2str(size(Ron)), ']']));
    end
    
    % Extract dimensions
    Nele = size(Ron, 1);
    Nbin = size(Ron, 3);
    
    % Verify that the off matrix is full-rank
    offranks = zeros(Nbin, 1);
    for b = 1:Nbin
        offranks(b) = rank(Roff(:,:,b));
    end
    num_sing_off = sum(offranks == 0);
    if num_sing_off > 0
        error('Off-source Covariance matrix must be full rank!\n');
    end
    
    % Compute steering vectors
    v = zeros(Nele, Nbin);
    for b = 1:Nbin
        [v(:,b),lambda] = eigs(Ron(:,:,b), Roff(:,:,b), 1);
        v(:,b) = sqrt(lambda)*v(:,b);
    end

end

function [ w ] = get_maxSNR_weights( v, Roff )
%GET_MAXSNR_WEIGHTS Formulates maximum signal-to-noise ratio beamformer
%weights given steering vectors and off-pointing covariance matrices
%   Detailed explanation goes here

    % Make sure the matrix product is valid
    if size(Roff, 1) ~= size(v, 1)
        error(sprintf(['MatVec product invalid\n',...
               '\t size(Roff)  = [', num2str(size(Roff)),  ']\n',...
               '\t size(v)     = [', num2str(size(v)),     ']']));
    end
    
    % Verify same number of frequency bins
    if size(Roff, 3) ~= size(v,2)
        error(sprintf(['Number of frequency channels does not match!\n',...
               '\t size(Roff,3)  = [', num2str(size(Roff,3)),  ']\n',...
               '\t size(v,3)     = [', num2str(size(v,3)),     ']']));
    end
        
    
    % Make sure that the off-pointing matrix is square
    if size(Roff, 1) ~= size(Roff, 2)
        error(sprintf(['Covariance matrix must be square!\n',...
               '\t size(Roff) = [', num2str(size(Roff)), ']']));
    end
    
    % Extract dimensions
    Nele = size(Roff, 1);
    Nbin = size(Roff, 3);
    
    % Verify that the off matrix is full-rank
    offranks = zeros(Nbin, 1);
    for b = 1:Nbin
        offranks(b) = rank(Roff(:,:,b));
    end
    num_sing_off = sum(offranks == 0);
    if num_sing_off > 0
        error('Off-source Covariance matrix must be full rank!\n');
    end
    
    % Compute weights
    w = zeros(Nele, Nbin);
    for b = 1:Nbin
        w(:,b) = Roff(:,:,b)\v(:,b);
        % w(:,b) = w(:,b)./norm(w(:,b));
        w(:,b) = w(:,b)/abs(w(:,b)'*v(:,b));
    end

end
