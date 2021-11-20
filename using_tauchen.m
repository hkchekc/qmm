N = 2;
sigma = .012;
rho = .9;
mu = -sigma^2/2/(1-rho);
w = 0.5 + rho/4;
sigmaZ = sigma/sqrt(1-rho^2);
baseSigma = w*sigma+(1-w)*sigmaZ;
[Z, Aprob] = tauchenhussey(N,mu,rho,sigma,baseSigma);
A = exp(Z);
Ni = 3;
sigmai = sqrt(.015);
mui = 0;
rhoi = .95;
wi = 0.5 + rhoi/4;
sigmaZi = sigmai/sqrt(1-rhoi^2);
baseSigmai = wi*sigmai+(1-wi)*sigmaZi;
[X, Yprob] = tauchenhussey(Ni,mui,rhoi,sigmai,baseSigmai);
Y = exp(X);
bigTrans = kron(Aprob, Yprob);
fileID = fopen('./krusell_smith/input/markov.txt','w');

fprintf(fileID,'%5d %5d %5d %5d %5d %5d\n', bigTrans);
fclose(fileID);
fileID = fopen('./krusell_smith/input/amarkov.txt','w');

fprintf(fileID,'%5d %5d\n', Aprob);
fclose(fileID);
fileID = fopen('./krusell_smith/input/ymarkov.txt','w');

fprintf(fileID,'%5d %5d %5d\n', Yprob);
fclose(fileID);