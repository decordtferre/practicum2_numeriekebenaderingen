function [beta, step] = GD(A, b)

    [N, n] = size(A);
    beta   = zeros(n, 1);
    tol    = 1e-6;
    gamma = 1/3;
    q  = 0.5;

    for k = 1:10000
        p    = 1 ./ (1 + exp(-b .* (A * beta)));
        grad = -(1/N) * (A' * ((1 - p) .* b));
        
        % vroeger stoppen als als de gradient reeds klein genoeg is
        % = convergentiecriterium
        if norm(grad) < tol
            break
        end
        

        f_k_1 = (1/N) * sum(log(1 + exp(-b .* (A * beta))));
        % zoekrichting bepalen
        p_k = -grad;

        step = 1;
        while true
            beta_k = beta + step * p_k;
            f_k = (1/N) * sum(log(1 + exp(-b .* (A * beta_k))));
            
            % fout vergelijken
            if f_k <= f_k_1 + gamma * step * (grad' * p_k)
                break
            end
            step = q * step;
            
            % stop bij te kleine stapgrootte
            if step < 1e-10
                break
            end
        end

        beta = beta + step * p_k;
        
    end
end