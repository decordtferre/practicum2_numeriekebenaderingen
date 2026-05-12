function [x, step] = GD(func, Dfunc, x0, alpha, tol, max_iter)


    % default waarden mochten sommige argumenten niet worden ingevuld
    if nargin < 4
        alpha = 1;
    end

    % parameters voor backtracking
    gamma = 1e-4;
    q = 0.5;

    if nargin < 5
        tol = 1e-6;
    end

    if nargin < 6
        max_iter = 100000;
    end

    x = x0;

    for step = 1:max_iter
        grad = Dfunc(x); % bereken gradient
        p = -grad; % zoekrichting
        alpha_k = alpha; % startwaarde voor stapgrootte
        
        while true
            x_new = x + alpha_k*p;
            if func(x_new) <= func(x) + gamma*alpha_k*(grad'*p)
                break;
            end

            alpha_k = q*alpha_k;

            if alpha_k < 1e-12
                break
            end
        end

        x = x_new;

        %convergentiecriterium
        if norm(grad) < tol
            x = x_new;
            break;
        end
    end
end