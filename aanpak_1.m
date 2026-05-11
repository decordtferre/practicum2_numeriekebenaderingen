load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;
max_n = 20;

xr = zeros(0, 0);
yr = zeros(0, 0);
catr = zeros(0, 0);

xe = zeros(0, 0);
ye = zeros(0, 0);
cate = zeros(0, 0);

% Verdeling opstellen
for i = 1:length(a1)
    if randi([0 1]) == 1
        xr = [xr; x(i)];
        yr = [yr; y(i)];
        catr = [catr; cat(i)];
    else
        xe = [xe; x(i)];
        ye = [ye; y(i)];
        cate = [cate; cat(i)];
    end
end

% Trainen en testen
for n=1:max_n
    %Trainen model
    A       = build_A(xr, yr, n);
    beta    = GD(A, catr);

    %Testdata evalueren
    A_e = build_A(xe, ye, n);
    b_hat   = classify(A_e, beta);
    fouten(n) = sum(b_hat ~= cate);

    CV(n) = 1/(2*size(xe,1))*sum(abs(cate-b_hat));
end

plot(1:max_n, CV, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('n'); ylabel('Kruisvalidatiefout');
grid on;

disp(fouten)