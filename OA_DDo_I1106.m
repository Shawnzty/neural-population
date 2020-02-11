%I-population Ne_1=1000 and E-population Ne_2=4000 
%Each neuron is nicely modified theta model 

%proberbility of synaptic connections are 0.2
%synaptic coupling is averaged. So each neuron in population 
%receives the same stimulus .

%XX_G are synaptic dynamics, v_1(phase),v_2(phase) are membrance potentials
%note that phase 0 means 0 rad. Phase "nx/2" means Pi rad 
%(This is different from other programs)

%dt=0.01ms num is total number of time step.(therefore, here is a 0.3s
%simulation in total) 
%Raster plot and firing rate are plotted
%Red in figure means excitatory neurons

for repeat=0:0

    %% Parameters
sss=1; % loop times
local=1; % no use
%0 for cut

Ne_1=1000; %I % 1000
% Ne_2=10000; %E
meanrate = [];
inout=[];
dt = 0.01; %0.005 0.0025
%a_master=2 %+ (floor(repeat/10)-5)*0.1;%1212
%a_I=a_master;
a_E=2;%2 % no use
sig = 0%0  ;%sqrt(2*D);% noise
num=60000;%120000 % total time

threshold=0.15;
u_end=0;
%for w=1 %=mu
theta=pi; % threshold of spike or not spike
V_R=-62;
V_T=-55;
I_V_syn=-70;
E_V_syn=0;
c_1=2/(V_T-V_R);

I_g_L=0.1;
E_g_L=0.08;%0.05 no use

II_c_2=(2*I_V_syn-V_T-V_R)/(V_T-V_R); % c2
IE_c_2=(2*E_V_syn-V_T-V_R)/(V_T-V_R);
EI_c_2=(2*I_V_syn-V_T-V_R)/(V_T-V_R); % no use
EE_c_2=(2*E_V_syn-V_T-V_R)/(V_T-V_R); % no use


t_1 = 0; %delay step to transmit
%II_t_r = 0.5; %refractory step 20 %0.5-parameter 
II_t_d= 5; % tau=5ms
II_t_r=0.5; ;%tau for inhibit
%II_t_r=(floor(repeat/10)+1)*0.1; ;%tau for inhibit

II_gbar = 0.138062206; % GABA on interneuron
II_rate = 50 %50 % expectation connection number p*N

% no use
IE_t_r = 0.5;
IE_t_d= 2;
IE_gbar = 0.010400214; % 0; 0.010400214;
IE_rate = 800;
EI_t_r = 0.5; 
EI_t_d= 5;%tau for inhibit
EI_gbar = 0.172577757; %0.172577757; 0;
EI_rate = 200;
EE_t_r = 0.5; 
EE_t_d= 2;
EE_gbar = 0.01291816; %0.01291816; 0; 
EE_rate = 800;


II_c_3=-1/II_t_r/II_t_d; % no use
II_c_5=II_gbar*II_rate/II_t_r/II_t_d; % no use


II_c_3e=-1/II_t_d; % -1/tao
II_c_4=-(II_t_r+II_t_d)/II_t_r/II_t_d; % no use

    a_o_n = (   2.9*10^(-4)  )  *10^6  ;% area of neuron [cm^2 * 10^6]
    gsyn_peak_SN(1,1) = 6.2; % g peak
    II_c_5e = gsyn_peak_SN(1,1)/a_o_n *II_rate; % mu=gpeak*p*N

    
% no use
IE_c_3=-1/IE_t_r/IE_t_d;
IE_c_4=-(IE_t_r+IE_t_d)/IE_t_r/IE_t_d;
IE_c_5=IE_gbar*IE_rate/IE_t_r/IE_t_d;
EI_c_3=-1/EI_t_r/EI_t_d;
EI_c_4=-(EI_t_r+EI_t_d)/EI_t_r/EI_t_d;
EI_c_5=EI_gbar*EI_rate/EI_t_r/EI_t_d;
EE_c_3=-1/EE_t_r/EE_t_d;
EE_c_4=-(EE_t_r+EE_t_d)/EE_t_r/EE_t_d;
EE_c_5=EE_gbar*EE_rate/EE_t_r/EE_t_d;


% no use
EE_G= zeros(2,num);
EI_G= zeros(2,num);
IE_G= zeros(2,num);
II_G= zeros(2,num+2);
%tmp_I2_1= 0;
%diff_I2_1=0;
%tmp_diff_I2_1=0;

%tmp_I2_2= 0;
%diff_I2_2=0;
%tmp_diff_I2_2=0;

    %% Input current
K = -0.0; %-0.3 % no use
eta_zero_N=2 %2 % average input current
del_N=0.1 %0.1 % larger range
    y_temp = transpose( (1:Ne_1)/(Ne_1+1));
        
    % Normal Lorentz
    eta_temp = del_N * tan (  pi * (y_temp - 1/2)  ) + eta_zero_N ;
   
% no use    
% R = 0.1 *power(10,1);
%S=[0.5*rand(Ne,Ne)];
for i=1:Ne_1
for j=1:Ne_1
    tmp = rand(1);
    if tmp > -5; %0.9; 
        S(i,j)=1;
    else S(i,j)=0;
    end
end
end

%S=[ones(Ne,Ne)];
    %% Initial values
v_1=2*pi*rand(Ne_1,1)-pi;    % Initial values of v % uniform distribution
%v_2=2*pi*rand(Ne_2,1)-pi;    % Initial values of v

    %% Container preparation
firings_1=[];             % spike timings
%firings_2=[];             % spike timings
v_all=zeros(num,Ne_1); % store all v_1 data
A_1 = zeros(num,1);
%A_2 = zeros(num,1);
out1 = zeros(num,1);
%out2 = zeros(num,1);
Nave=500;
in = zeros(num,1); % no use
c=0.0; % no use

I2_1=zeros(num,1); % no use
%I2_2=zeros(num,1);

kk=1;
tmp_pre=0;

    %% Simulation
for k=1:num    % count for all neurons    
%  I_common= sigma * randn(Ne_1,1); % only for noisesyn 
  I_common= randn(1,1); % only for noisesyn % no use
  I_1= sig*randn(Ne_1,1); % input with noise
  In(k) = 0.0;
  %In(t) = 0.5 * sin(2 * pi * f * t / 2000);
%  I_1= sigma * (sqrt(1-c) * I_1 + sqrt(c) * I_common);%1/sqrt(2) * (I_1 + I_common);
%   I_1= I_1 + In(i);
  if k>0
  fired_1=find(v_1>=theta);   % indices of spikes
  A_1(k)=size(fired_1,1)/dt / Ne_1;
  firings_1=[firings_1; k+0*fired_1,fired_1]; %i-1= actual firing time
 v_1(fired_1)=v_1(fired_1)-2*pi;
% ref_1 = find(firings_1(:,1)>(i-1-tauR));
% v_1(firings_1(ref_1,2))=0;
  end
  
%   I_2= sig*randn(Ne_2,1); % input
%   In(k) = 0.0; 
%   %In(t) = 0.5 * sin(2 * pi * f * t / 2000);
% %  I_2=  sigma * (sqrt(1-c) * I_2 + sqrt(c) * I_common);%1/sqrt(2) * (I_2 + I_common);
% %  I_2= I_2 + In(i);
% if k>1  
% fired_2=find(v_2>=theta);    % indices of spikes
%   A_2(k)=size(fired_2,1)/dt / Ne_2;
%   firings_2=[firings_2; k+0*fired_2,fired_2]; %i-1= actual firing time
%  v_2(fired_2)=v_2(fired_2)-2*pi;
% % ref_2 = find(firings_2(:,1)>(i-1-tauR));
% % v_2(firings_2(ref_2,2))=0;
% end
%  
 %  feedback = find(firings(:,1)==(i-5)); 
%  I2 =  sum(S(:,firings(feedback,2)),2);
  if k > t_1
    II_G(1,k+1)= II_G(1,k)+ II_c_3e*dt*II_G(1,k)+II_c_5e*A_1(k-t_1)*dt;%flux no need eta 1st
    
%        II_G(1,k+1)= II_G(1,k)+ dt*II_G(2,k);
%    II_G(2,k+1)=II_c_3*dt*II_G(1,k)+(1+II_c_4*dt)*II_G(2,k)+II_c_5*A_1(k-t_1)*dt;%flux no need

%     IE_G(1,k+1)= IE_G(1,k)+ dt*IE_G(2,k);
%     IE_G(2,k+1)=II_c_3*dt*IE_G(1,k)+(1+IE_c_4*dt)*IE_G(2,k)+IE_c_5*A_2(k-t_1)*dt;
%     EI_G(1,k+1)= EI_G(1,k)+ dt*EI_G(2,k);
%     EI_G(2,k+1)=II_c_3*dt*EI_G(1,k)+(1+EI_c_4*dt)*EI_G(2,k)+EI_c_5*A_1(k-t_1)*dt;
%     EE_G(1,k+1)= EE_G(1,k)+ dt*EE_G(2,k);
%     EE_G(2,k+1)=II_c_3*dt*EE_G(1,k)+(1+EE_c_4*dt)*EE_G(2,k)+EE_c_5*A_2(k-t_1)*dt;
  end

  v_1=v_1+(-I_g_L * cos(v_1) + c_1 * (1+cos(v_1)).*(eta_temp - c_1*sig*sig/2*sin(v_1)) +II_G(1,k)*(II_c_2 * (1+cos(v_1))-sin(v_1)) +IE_G(1,k)*(IE_c_2 * (1+cos(v_1))-sin(v_1)))*dt + c_1 * (1+cos(v_1)).* I_1 * sqrt(dt); 
  % eta
  
strage(sss)=v_1(900);
sss=sss+1;

%  v_2=v_2+(-E_g_L * cos(v_2) + c_1 * (1+cos(v_2)).*(a_E - sig*sig/2*sin(v_2)) +EI_G(1,k)*(EI_c_2 * (1+cos(v_2))-sin(v_2)) +EE_G(1,k)*(EE_c_2 * (1+cos(v_2))-sin(v_2)))*dt + c_1 * (1+cos(v_2)).* I_2 * sqrt(dt); 
  if(k==20000) real_threshold = mean(II_G(1,1:20000));
  end
      if(k>=20000)
        if(II_G(1,k)>threshold)&&(II_G(1,k)<=II_G(1,k-1))&&(II_G(1,k+1)<II_G(1,k-1))&&(II_G(1,k-1)>II_G(1,k-2))&&(II_G(1,k-1)>II_G(1,k-3))&&(k>tmp_pre+1000)
%         if(II_G(1,k)<real_threshold)&&(II_G(1,k+1)>=real_threshold)&&(k>tmp_pre+1000)
        %u_start=u_end(kk);
        u_end(kk)=k;
        tmp_pre=k;
        kk=kk+1;
%        u_end-u_start
        end
      end
  v_all(k,:)=v_1; % store all the v_1 data
 
end

%period=(u_end(kk-1)-u_end(1))/(kk-2);
%2*pi/(period*0.01/1) %0.01=dt, with '/1000'->unit=[rad/s]

% no use
for i=1:num
    if i>Nave
    out1(i-(Nave)/2) = mean(A_1(i-Nave:i));
 %   out2(i-(Nave)/2) = mean(A_2(i-Nave:i));
    end
end

    %% Plot
 subplot(3,1,1);
 plot(firings_1(:,1),firings_1(:,2),'.');
 xlabel('time(s)');
 ylabel('Number of firing neurons');
 %plot(firings_1(:,1)/100/1000,firings_1(:,2),'.');
 str = strcat(num2str(Ne_1),{' Neurons with excitability ('},num2str(eta_zero_N),{', '},num2str(del_N),{') and noise '},num2str(sig));
 title(str)
 
 subplot(3,1,2);
 plot(+II_G(1,:));
 axis([0 num 0 1]);
 xlabel('time(s)');
 ylabel('Synaptic conductance');
 
 subplot(3,1,3);
 hist(v_1,50);
 xlabel('Membrane potential (mV)');
 ylabel('Number of neurons');


%plot(In);
%meanrate(w+1)=size(firings,1)/Ne/(T*dt);
meanrate = [meanrate; mean(out1(1000:10000))];

%tmp = corrcoef(out1(1000:10000),In(1000:10000))
%inout = [inout; tmp(1,2)];
VI_time=(V_R+V_T)/2+(V_T-V_R)/2*tan(strage/2);
%VI_time=(V_R+V_T)/2+(V_T-V_R)/2*(sin(strage)./(1+cos(strage)+4*10^(-4)));%not so good
% plot(VI_time);axis([0 120000 -100 200]);
end
%end