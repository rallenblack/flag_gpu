function [ w ] = get_maxSNR_weights( v, Roff )
%GET_MAXSNR_WEIGHTS Formulates maximum signal-to-noise ratio beamformer
%weights given steering vectors and off-pointing covariance matrices
%   Detailed explanation goes here

    % Make sure the matrix product is valid
    if size(Roff, 1) ~= size(v, 1)
        error(['MatVec product invalid\n',...
               '\t size(Roff)  = [', num2str(size(Roff)),  ']\n',...
               '\t size(v)     = [', num2str(size(v)),     ']']);
    end
    
    % Verify same number of frequency bins
    if size(Roff, 3) ~= size(v,2)
        error(['Number of frequency channels does not match!\n',...
               '\t size(Roff,3)  = [', num2str(size(Roff,3)),  ']\n',...
               '\t size(v,3)     = [', num2str(size(v,3)),     ']']);
    end
        
    
    % Make sure that the off-pointing matrix is square
    if size(Roff, 1) ~= size(Roff, 2)
        error(['Covariance matrix must be square!\n',...
               '\t size(Roff) = [', num2str(size(Roff)), ']']);
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
        w(:,b) = w(:,b)./norm(w(:,b));
    end

end

