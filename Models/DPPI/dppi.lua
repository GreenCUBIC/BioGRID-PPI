--[[
    ---------- Original Work this file is based on: ----------
    Published in Bioinformatics journal featuring ISMB/ECCB 2019
    Title: 'Predicting protein–protein interactions through sequence-based deep learning'
    Authors: Somaye Hashemifar, Behnam Neyshabur, Aly A. Khan and Jinbo Xu
    Journal: Bioinformatics
    Volume: 34
    Number: 
    Pages: i802-i810
    Year: 2018
    Publisher: Oxford University Press
    DOI: 10.1093/bioinformatics/bty573
    git: https://github.com/hashemifar/DPPI
    
    ---------- This file ----------
    This dppi.lua file is a combining and modification from the original git files.
    Main modifications include a change of command-line argument usage for execution and a choice of cross-validation 
    or a single train/test split. Prediction probabilities of each interaction in test data are also saved to file.
    Author: Eric Arezza
    Last Updated: February 8, 2021
    
    Description:
        Deep learning CNN approach to binary classification of protein-protein interaction prediction.

]]

--------------------------DPPI------------------------------------------------------
require 'torch'
require 'cutorch'
require 'paths'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'math'

---------------------MAIN-INITIALIZE-------------------------------------------------
print('==> Options')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')

-- global:
cmd:option('-device', 1, 'set GPU Device')
cmd:option('-string', 'A', 'suffix to log files')
cmd:option('-saveModel', false, 'saves the model if true')
cmd:option('-seed', 1, 'manual seed')
cmd:option('-dataDir', './', 'directory path for data files')

-- data:
cmd:option('-train','A','data to use for training')
cmd:option('-test','A','data to use for testing')

-- training params:
cmd:option('-string', 'A', 'suffix to log files')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ADAM')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 100, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-epochs', 100, 'number of epochs')
cmd:option('-epochID', 1, 'staring epoch -used for resuming the run on servers')
cmd:option('-less_eval', false, 'evaluate every 10 epochs')
cmd:option('-crop', true, 'crop the sequence if true')
cmd:option('-cropLength', 512, 'length of the cropped sequence')
cmd:text()
opt = cmd:parse(arg or {})
for k, v in pairs(opt) do
   print (k, v)
end

cutorch.setDevice(opt.device)

-- The string used to save the model and logs
---saveName = string.format("%s-%s-%s_crop%d-%s-rate_%g", opt.string, opt.train, opt.test, opt.cropLength, opt.optimization, opt.learningRate)
saveName = string.format("%s-%s-%s", opt.string, opt.train, opt.test)

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

if file_exists(opt.train..'.csv') == false or file_exists(opt.train..'.node') == false then
    --exit program
    os.exit()
end

---------------------MAIN-INITIALIZE----------------------------------

---------------------PREPROCESSING-------------------------------------------
------------------------CSV--------------------------------------------------
-- class to manage comma separate value file + two directly-usable functions
-- various function to manage csv files
-- these CSV files all have a comma delimiter and use " as the quote character
-- ref:
-- http://www.lua.org/pil/20.4.html
-- http://www.torch.ch/manual/torch/utility#torchclass
print('############# PROCESSING RAW DATA #############')
print('==> defining .csv input')
do
    -- create class Csv
    local Csv = torch.class("Csv")
    
    -- initializer
    function Csv:__init(filepath, mode, char)
       local msg = nil
       self.file, msg = io.open(filepath, mode)
       self.separator = char or ','
       if not self.file then error(msg) end
    end

    -- close underlying file
    function Csv:close()
       io.close(self.file)
    end
    
    -- return iterator that reads all the remaining lines
    function Csv:lines()
       return self.file:lines()
    end
    
    -- return next record from the csv file
    -- return nill if at end of file
    function Csv:read() 
       local line = self.file:read()
       if not line then return nil end
       return fromcsv(line, self.separator)
    end
    
    -- return all records as an array
    -- each element of the array is an array of strings
    -- should be faster than reading record by record
    function Csv:readall()
       local all = self.file:read("*all")
       local res = {}
       for line in string.gmatch(all, "([^\n]*)\n") do
        res[#res+1] = fromcsv(line, self.separator)
       end
       return res
    end

    -- write array of strings|numbers to the csv file followed by \n
    -- convert to csv format by inserting commas and quoting where necessary
    -- return nil
    function Csv:write(a)
       res, msg = self.file:write(tocsv(a, self.separator),"\n")
       if res then return end
       error(msg)
    end
    
    -- the next 3 functions came from
    -- http://www.lua.org/pil/20.4.html
    
    -- static method
    -- convert an array of strings or numbers into a row in a csv file
    function tocsv(t, separator)
       local s = ""
       for _,p in pairs(t) do
        s = s .. separator .. escapeCsv(p, separator)
       end
       return string.sub(s, 2) -- remove first comma
    end
       
    -- private 
    -- enclose commas and quotes between quotes and escape original quotes
    function escapeCsv(s, separator)
       if string.find(s, '["' .. separator .. ']') then
       --if string.find(s, '[,"]') then
        s = '"' .. string.gsub(s, '"', '""') .. '"'
       end
       return s
    end

    -- break record from csv file into array of strings
    function fromcsv(s, separator)
       if not s then error("s is null") end
       s = s .. separator -- end with separator
       local t = {}
       local fieldstart = 1
       repeat
        -- next field is quoted? (starts with "?)
       if string.find(s, '^"', fieldstart) then
       local a, c
       local i = fieldstart
       repeat
       -- find closing quote
       a, i, c = string.find(s, '"("?)', i+1)
       until c ~= '"'  -- quote not followed by quote?
       if not i then error('unmatched "') end
       local f = string.sub(s, fieldstart+1, i-1)
       table.insert(t, (string.gsub(f, '""', '"')))
       fieldstart = string.find(s, separator, i) + 1
       else
       local nexti = string.find(s, separator, fieldstart)
       table.insert(t, string.sub(s, fieldstart, nexti-1))
       fieldstart = nexti + 1
       end
       until fieldstart > string.len(s)
       return t
    end
end
-----------------------CSV---------------------------------------------------

---------------------CONVERT-CSV-TO-DAT--------------------------------------
print('==> creating _labels.dat from .csv')
train_ppiFile = Csv(opt.train..'.csv', "r")
train_ppi = train_ppiFile:readall()
train_ppiFile:close()
torch.save( opt.train..'_labels.dat', train_ppi)

if opt.test ~= opt.train then
    test_ppiFile = Csv(opt.test..'.csv', "r")
    test_ppi = test_ppiFile:readall()
    test_ppiFile:close()
    torch.save( opt.test..'_labels.dat', test_ppi)
end
---------------------CONVERT-CSV-TO-DAT--------------------------------------

---------------------CREATE-CROP---------------------------------------------
print('==> creating features _profile_crop.t7 and _number_crop.t7 from .node')
crop_size = 512;
NumFeatures = 20
C = 0.8

background = {}

background[1]=0.0799912015849807; --A
background[2]=0.0484482507611578; --R
background[3]=0.044293531582512; --N
background[4]=0.0578891399707563; --D
background[5]=0.0171846021407367; --C
background[6]=0.0380578923048682; --Q
background[7]=0.0638169929675978; --E
background[8]=0.0760659374742852; --G
background[9]=0.0223465499452473; --H
background[10]=0.0550905793661343; --I
background[11]=0.0866897071203864; --L
background[12]=0.060458245507428; --K
background[13]=0.0215379186368154; --M
background[14]=0.0396348024787477; --F
background[15]=0.0465746314476874; --P
background[16]=0.0630028230885602; --S
background[17]=0.0580394726014824; --T
background[18]=0.0144991866213453; --W
background[19]=0.03635438623143; --Y
background[20]=0.0700241481678408; --V

if file_exists(opt.train..'_profile_crop_'..crop_size..'.t7') and file_exists(opt.train..'_number_crop_'..crop_size..'.t7') then
    print(opt.train..' crop files found')
else
local proteinFile = Csv(opt.train..".node","r")
local proteinString = proteinFile:readall()

ppFeature = {}
pNumber = {}

for i=1, #proteinString do
  
  local fileName = opt.train..'/'..proteinString[i][1]
  if file_exists( fileName ) then
    local proFile = Csv( fileName, 'r', '\t')
    local profile = proFile:readall()
   
    pNumber[ proteinString[i][1] ] = math.ceil( 2 * #profile / crop_size  - 1)

    if pNumber[ proteinString[i][1] ] < 1 then
      pNumber[ proteinString[i][1] ] = 1
    end
   
    for c = 1, pNumber[ proteinString[i][1] ] do
      start = math.min( (c-1) * crop_size / 2 + 1, #profile - crop_size + 1 )
      ppFeature[ proteinString[i][1]..'-sub'..c ] = torch.Tensor(1, 20, crop_size, 1):zero() 
      if start > 0 then
        for j=start,start+crop_size-1 do
          for k=1,20 do
            ppFeature[ proteinString[i][1]..'-sub'..c ][1][k][j-start+1] = math.log(C*( tonumber(profile[j][k+20]) / 100 )+(1-C)*background[k])  
          end
        end
      else
        for j=1,#profile do
          for k=1,20 do
            ppFeature[ proteinString[i][1]..'-sub'..c ][1][k][j-math.floor(start/2)] = math.log(C*( tonumber(profile[j][k+20]) / 100 )+(1-C)*background[k])
          end
        end
      end
    end 
    proFile:close()
  end
end

proteinFile:close()
collectgarbage()
torch.save(opt.train..'_profile_crop_'..crop_size..'.t7', ppFeature )
torch.save(opt.train..'_number_crop_'..crop_size..'.t7', pNumber )
end

if opt.test ~= opt.train then
if file_exists(opt.test..'_profile_crop_'..crop_size..'.t7') and file_exists(opt.test..'_number_crop_'..crop_size..'.t7') then
    print(opt.test..' crop files found')
else
local proteinFile = Csv(opt.test..".node","r")
local proteinString = proteinFile:readall()

ppFeature = {}
pNumber = {}

for i=1, #proteinString do
  
  local fileName = opt.test..'/'..proteinString[i][1]
  if file_exists( fileName ) then
    local proFile = Csv( fileName, 'r', '\t')
    local profile = proFile:readall()

    pNumber[ proteinString[i][1] ] = math.ceil( 2 * #profile / crop_size  - 1)

    if pNumber[ proteinString[i][1] ] < 1 then
      pNumber[ proteinString[i][1] ] = 1
    end
   
    for c = 1, pNumber[ proteinString[i][1] ] do
      start = math.min( (c-1) * crop_size / 2 + 1, #profile - crop_size + 1 )
      ppFeature[ proteinString[i][1]..'-sub'..c ] = torch.Tensor(1, 20, crop_size, 1):zero() 
      if start > 0 then
        for j=start,start+crop_size-1 do
          for k=1,20 do
            ppFeature[ proteinString[i][1]..'-sub'..c ][1][k][j-start+1] = math.log(C*( tonumber(profile[j][k+20]) / 100 )+(1-C)*background[k])  
          end
        end
      else
        for j=1,#profile do
          for k=1,20 do
            ppFeature[ proteinString[i][1]..'-sub'..c ][1][k][j-math.floor(start/2)] = math.log(C*( tonumber(profile[j][k+20]) / 100 )+(1-C)*background[k])
          end
        end
      end
    end 
    proFile:close()
  end
end

proteinFile:close()
collectgarbage()

torch.save(opt.test..'_profile_crop_'..crop_size..'.t7', ppFeature )
torch.save(opt.test..'_number_crop_'..crop_size..'.t7', pNumber )
end
end
print('==> raw data to representation processing completed')
---------------------CREATE-CROP---------------------------------------------
--------------------PREPROCESSING--------------------------------------------

--------------------LOAD-DATA-----------------------------------------
function pair_crop_load(file, pos_weight, pNumber)
print ('==> loading data '..file)

  local original_data = torch.load(file)

  local crop_data = {}

  counter = 1
  for i=1,#original_data do

   for j=1,pNumber[ original_data[i][1] ] do
      for k=1,pNumber[ original_data[i][2] ] do
    	crop_data[ counter ] = {}
        crop_data[ counter ][1] = original_data[i][1]..'-sub'..j
        crop_data[ counter ][2] = original_data[i][2]..'-sub'..k
        crop_data[ counter ][3] = original_data[i][3]
        counter = counter + 1
      end
    end
  end
  local Data = {
        data = crop_data,
        org_data = original_data,
        pNum = pNumber,
        size = 0,
      }
  Data.size = #Data.data
  Data.pos_weight = pos_weight
  return Data
end

function pair_seq_load_batch( Data, index, feature )

    -- Tensor of labels
    targets = torch.CudaTensor(index:size(1))

    -- Tensor of weights
    weights = torch.CudaTensor(index:size(1))

  inputs = {}
  inputs[1]={}
  inputs[2]={}
  for i=1, index:size(1) do

    local firstInd = 1
    local secondInd = 2
    if math.random() > 0.5 then
      firstInd = 2
      secondInd = 1
    end
    inputs[1][i] = feature[ Data.data[ index[i] ][ firstInd ] ]
    inputs[2][i] = feature[ Data.data[ index[i] ][ secondInd ] ]
      
    if num_outputs == 1 then
      targets[i] = 2 * Data.data[ index[i] ][ 3 ] - 1
    else
      targets[i] = Data.data[ index[i] ][ 3 ]
    end
  
    if targets[i] == 1 then
      weights[i] = Data.pos_weight
    else
      weights[i] = 1
    end

  end
  return inputs, targets, weights
end

-- List of validation sets and their names
valNames = {}
valData = {}

-- loading the datasets
if opt.train then 
  if opt.crop then
    pNumber = torch.load( opt.dataDir..opt.train..'_number_crop_'..opt.cropLength..'.t7' )
    trainData = pair_crop_load( opt.dataDir..opt.train..'_labels.dat',10, pNumber )
    print(#trainData.org_data..' interactions')
    print(trainData.size)
    
    if opt.test and opt.test ~= opt.train then
        test_pNumber = torch.load( opt.dataDir..opt.test..'_number_crop_'..opt.cropLength..'.t7' )
        testData = pair_crop_load( opt.dataDir..opt.test..'_labels.dat',10, test_pNumber )
        
        valNames[1] = opt.test
        valData[1] = pair_crop_load( opt.dataDir..opt.test..'_labels.dat',10, test_pNumber )
        print(#testData.org_data..' interactions')
        print(valData[1].size)
        test_feature = torch.load( opt.dataDir..opt.test..'_profile_crop_'..opt.cropLength..'.t7' )
        
        shuffle = torch.randperm( valData[1].size )
        for i=1, valData[1].size do
            shuffle[i] = i
        end
        test_inputs, test_targets, test_weights = pair_seq_load_batch(valData[1], shuffle, test_feature)
    else
        valNames[1] = 'validation'
        valData[1] = pair_crop_load( opt.dataDir..opt.train..'_valid_labels.dat',10, pNumber )
        print(valData[1].size)
    end
    
    train_feature = torch.load( opt.dataDir..opt.train..'_profile_crop_'..opt.cropLength..'.t7' )
    num_features = train_feature[ trainData.data[1][1] ]:size(2)
    num_outputs = 1
    
    shuffle = torch.randperm( trainData.size )
    for i=1, trainData.size do
        shuffle[i] = i
    end
    train_inputs, train_targets, train_weights = pair_seq_load_batch(trainData, shuffle, train_feature)

  end
else
  error( 'Unknown dataset!')
end
----------------------LOAD-DATA---------------------------------------------

-----------------------MODEL------------------------------------------------
print('==> creating the model') -- creating the model
function build_model()
    model = nn.Sequential()

    my_net = nn.Sequential()
    my_net:add( nn.JoinTable(1) )

    my_net:add( cudnn.SpatialConvolution( num_features , 64, 1, 5, 1, 1, 0, 2) )
    my_net:add( cudnn.SpatialBatchNormalization( 64 ) )
    my_net:add( cudnn.ReLU(true) )
    my_net:add( cudnn.SpatialAveragePooling( 1, 4) )

    my_net:add( cudnn.SpatialConvolution( 64 , 128, 1, 5, 1, 1, 0, 2) )
    my_net:add( cudnn.SpatialBatchNormalization( 128 ) )
    my_net:add( cudnn.ReLU(true) )
    my_net:add( cudnn.SpatialAveragePooling( 1, 4) )

    my_net:add( cudnn.SpatialConvolution( 128 , 256, 1, 5, 1, 1, 0, 2) )
    my_net:add( cudnn.SpatialBatchNormalization( 256 ) )
    my_net:add( cudnn.ReLU(true) )
    my_net:add( cudnn.SpatialAveragePooling( 1, 4) )

    my_net:add( cudnn.SpatialConvolution( 256 , 512, 1, 5, 1, 1, 0, 2) )
    my_net:add( cudnn.SpatialBatchNormalization( 512 ) )
    my_net:add( cudnn.ReLU(true) )

    my_net:add( cudnn.SpatialAveragePooling( 1, opt.cropLength/64) )
    my_net:add( nn.Reshape(512) )

    outH = 512
    lin_net1 = nn.Sequential()
    lin_net1:add( nn.Linear( outH, outH) )
    lin_net1:add( nn.BatchNormalization( outH,1e-5,0.1,false ) )
    lin_net1:add( nn.ReLU(true))

    lin_net2 = nn.Sequential()
    lin_net2:add( nn.Linear( outH, outH) )
    lin_net2:add( nn.BatchNormalization( outH,1e-5,0.1,false ) )
    lin_net2:add( nn.ReLU(true))

    lin_net1c = lin_net1:clone('weight','bias','gradWeight','gradBias')
    lin_net2c = lin_net2:clone('weight','bias','gradWeight','gradBias')
      
    rand_layer1 = nn.Sequential()
    rand_layer1:add( nn.ConcatTable():add( lin_net1 ):add( lin_net2 ) )
    rand_layer1:add( nn.JoinTable(2) )
    rand_layer1:add( nn.Reshape(1024) )
  
    rand_layer2 = nn.Sequential()
    rand_layer2:add( nn.ConcatTable():add( lin_net2c ):add( lin_net1c ) )
    rand_layer2:add( nn.JoinTable(2) )
    rand_layer2:add( nn.Reshape(1024) )

    model:add( nn.MapTable():add( my_net ) )

    lin1 = nn.Linear( 512, 1);
    lin2 = lin1:clone('weight','bias','gradWeight','gradBias')

    model:add( nn.ParallelTable():add( rand_layer1  ):add( rand_layer2 ) )
    model:add( nn.CMulTable() )
    model:add( nn.View( 1024 ) )
    model:add( nn.Linear( 1024, 1 )) 
    model:add( nn.View(1) )    

    if num_outputs == 1 then
        model:add( nn.Sigmoid() )
        --model:add(nn.SoftMax())
    end
    print(model)
    return model
end

-- Initialization
function ConvInit(name)
   for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if cudnn.version >= 4000 then
         v.bias = nil
         v.gradBias = nil
      else
         v.bias:zero()
      end
   end
end

function BNInit(name)
   for k,v in pairs(model:findModules(name)) do
      v.weight:fill(1)
      v.bias:zero()
   end
end

model = build_model()

ConvInit('cudnn.SpatialConvolution')
ConvInit('nn.SpatialConvolution')
BNInit('fbnn.SpatialBatchNormalization')
for k,v in pairs(model:findModules('nn.Linear')) do
   v.bias:zero()
end

-- loss function
 criterion = nn.BCECriterion()

-- switch to Cuda
model:cuda()
criterion:cuda()
----------------------MODEL-------------------------------------------------

----------------------TRAIN-------------------------------------------------
print ('==> defining training procedure')
-- choosing the optimization method and hyper-parameters
if opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 0
   }
   optimMethod = optim.sgd
elseif opt.optimization =='ADAM' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
   }
   optimMethod = optim.adam
else
   error('unknown optimization method')
end
  
function train()
  parameters,gradParameters = model:getParameters()
  epoch = epoch or 1

  if epoch == 300 or epoch == 400 then
    opt.learningRate = opt.learningRate / 10
  end
  
  local time = sys.clock()
  
  model:training()
  
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local train_obj = 0
  local shuffle = torch.randperm( trainData.size )
  local num_batch = math.floor( trainData.size / opt.batchSize ) 

  for t = 1, num_batch*opt.batchSize, opt.batchSize do

    xlua.progress(t, num_batch*opt.batchSize )
    
    --load_batch( trainData, shuffle[{{t,t+opt.batchSize-1}}] )
    inputs, targets, weights = pair_seq_load_batch(trainData, shuffle[{{t,t+opt.batchSize-1}}], train_feature)
    model:zeroGradParameters()
    ---------------------------------------------
    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      local output = model:forward( inputs )
      local f = criterion:forward(output, targets)   
      train_obj = train_obj + f
      df_do = criterion:backward(output, targets)

      df_do:cmul(weights)
      model:backward(inputs, df_do)

      --Values of RP are fixed
      model.modules[2].modules[1].modules[1].modules[1].modules[1].gradWeight:zero()
      model.modules[2].modules[1].modules[1].modules[1].modules[1].gradBias:zero()
      model.modules[2].modules[1].modules[1].modules[2].modules[1].gradWeight:zero()
      model.modules[2].modules[1].modules[1].modules[2].modules[1].gradBias:zero()
          
      model.modules[2].modules[2].modules[1].modules[1].modules[1].gradWeight:zero()
      model.modules[2].modules[2].modules[1].modules[1].modules[1].gradBias:zero()
      model.modules[2].modules[2].modules[1].modules[2].modules[1].gradWeight:zero()
      model.modules[2].modules[2].modules[1].modules[2].modules[1].gradBias:zero()

      return f,gradParameters
    end
    
    optimMethod(feval, parameters, optimState)
  end

    time = sys.clock() - time
    time = time / ( num_batch * opt.batchSize )
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  epoch = epoch + 1
  train_obj = train_obj / ( num_batch * opt.batchSize )
  print('Objective: '.. train_obj )
  parameters, gradParameters = nil, nil

  collectgarbage()
  collectgarbage()
end

------------------------TRAIN----------------------------------------------
------------------------EVALUATE-------------------------------------------
print ('==> defining evaluation procedure')

-- evaluate the model on training set and all validation sets
function evaluateAll( epoch )
  print('################## EVALUATION ##################')
 
  prlog[epoch] = {}
  prlog[epoch][1] = {}
  prRank = false
  mylog[epoch][1], mylog[epoch][2],prlog[epoch][1][1],prlog[epoch][1][2], prlog[epoch][1][3] = evaluate( trainData, train_feature )
  print(' Training MAP and Objective ('..opt.train..' - '..trainData.size..') : '.. mylog[epoch][2] )

  local tmp_obj = 0 

  for i=1,#valData do

    prlog[epoch][1+i] = {}    
    if (i==1) then
    	prRank = false
    else
	prRank = true
    end
    tmp_obj, mylog[epoch][2+i],prlog[epoch][1+i][1],prlog[epoch][1+i][2], prlog[epoch][1+i][3] = evaluate( valData[i] , test_feature)
    print(' Validation MAP ('..valNames[i]..' - '..valData[i].size..') : '.. mylog[epoch][2+i] )
  end
end

-- evaluate the model on the given data
function evaluate( Data , feature)

  parameters,gradParameters = model:getParameters()
  model:evaluate()

  local val_obj = 0
  local counter = 0
  local shuffle = torch.randperm( Data.size )
  for i=1,Data.size do
    shuffle[i]=i
  end

  local num_batch = math.floor( Data.size / opt.batchSize )
  
  local val_score = torch.Tensor( num_batch*opt.batchSize, 1)
  local val_labels = torch.Tensor( num_batch*opt.batchSize, 1) 

  for t = 1, num_batch*opt.batchSize, opt.batchSize do
    inputs, targets, weights = pair_seq_load_batch( Data, shuffle[{{t,t+opt.batchSize-1}}] , feature)
    
    local output = model:forward( inputs )
    
    for k=1,output:size(1) do
      val_score[counter*opt.batchSize+k][1] = output[k][1]
      val_labels[counter*opt.batchSize+k][1] = targets[k]
    end
    counter = counter + 1
  end 

  val_obj = val_obj / counter
  if opt.crop then
    val_score, val_labels = recalculate_crop( val_score, val_labels, Data )
  end

  local val_map, precision, recall, specificity  = MAP( val_score, val_labels )
  parameters,gradParameters = nil, nil

  collectgarbage()
  collectgarbage()

  return val_obj, val_map, precision, recall, specificity
end

function recalculate_crop( v_score, v_labels, Data )

  new_score = torch.Tensor( #Data.org_data , 1):zero()
  new_labels = torch.Tensor( #Data.org_data , 1):zero()
  counter = 1

  predictions = {}

  for i=1, #Data.org_data do

    myscore1 = 0
    myscore2 = 0
    for j=1, Data.pNum[ Data.org_data[i][1] ] do
      for k=1, Data.pNum[ Data.org_data[i][2] ] do
        new_labels[i][1] = Data.org_data[i][3]
        if counter <= v_score:size(1) then
          myscore1 =  math.max( myscore1, v_score[counter][1] )
          myscore2 = myscore2 + v_score[counter][1]
          counter = counter + 1
        end
      end
    end
    new_score[i][1] = myscore1
    predictions[i] = Data.org_data[i][1]..'\t'..Data.org_data[i][2]..'\t'..tonumber(string.format("%.6f", myscore1))
  end
  
  return new_score, new_labels
end

-- Calculates the Mean Average Precision (MAP)
function MAP (score, truth)
  local x,ind,map,P,TP,FP,N  
  x, ind = torch.sort(score, 1, true)
  if num_outputs == 1 then
    truth:add(1):div(2)
  end
  
  P = torch.sum( truth,1 )
  local precision = torch.Tensor(score:size(1),1)
  local recall = torch.Tensor(score:size(1),1)
  local specificity = torch.Tensor(score:size(1),1)

  my_error = 0
  map = 0
  for c=1, score:size(2) do
    TP = 0
    FP = 0
    FN = 0

    N = score:size(1) - P[1][c] 
    for i=1, score:size(1) do  
      TP = TP + truth[ind[i][c]][c]
      FP = FP + (1 - truth[ind[i][c]][c] )      

      precision[i][1] = TP / (FP + TP)
      recall[i][1] = TP / P[1][c] 
      specificity[i][1] = FP / N  
      
      map = map + ( truth[ind[i][c]][c] * TP / ( P[1][c] * ( FP + TP ) ) )
    end
  end
  print("error ", my_error)
  map = map / ( score:size(2) ) 
  return map, precision, recall, specificity

end
--------------------EVALUATE---------------------------------------

----------------PREDICT--------------------
function make_prediction (Data, feature)

  local val_obj = 0
  local counter = 0
  local shuffle = torch.randperm( Data.size )
  for i=1,Data.size do
    shuffle[i]=i
  end

  local num_batch = math.floor( Data.size / opt.batchSize )
  
  local val_score = torch.Tensor( num_batch*opt.batchSize, 1)
  local val_labels = torch.Tensor( num_batch*opt.batchSize, 1) 

  for t = 1, num_batch*opt.batchSize, opt.batchSize do
    inputs, targets, weights = pair_seq_load_batch( Data, shuffle[{{t,t+opt.batchSize-1}}] , feature)
    
    local output = model:forward( inputs )
    
    for k=1,output:size(1) do
      val_score[counter*opt.batchSize+k][1] = output[k][1]
      val_labels[counter*opt.batchSize+k][1] = targets[k]
    end
    counter = counter + 1
  end 

  val_obj = val_obj / counter
  if opt.crop then
    val_score, val_labels = recalculate_crop( val_score, val_labels, Data )
  end

end
-------------PREDICT------------------
function get_performance(interactions, predicted)
    local tp = 0
    local fp = 0 
    local tn = 0 
    local fn = 0
    local total_pos = 0
    local total_neg = 0
    
    if #interactions ~= # predicted then
        print("Number of predictions not equal to .csv")
        return
    end
    
    for i=1, #predicted do
        --compare values
        actual = tonumber(interactions[i][3])
        pred = predicted[i][1]:split('\t')[3]
        pred = math.floor(tonumber(pred) + 0.5)
        --tally metrics
        if actual == 1 and pred == 1 then
            total_pos = total_pos + 1
            tp = tp + 1
        elseif  actual == 0 and pred == 0 then
            total_neg = total_neg + 1
            tn = tn + 1
        elseif actual == 1 and pred == 0 then
            total_pos = total_pos + 1
            fn = fn + 1
        elseif actual == 0 and pred == 1 then
            total_neg = total_neg + 1
            fp = fp + 1
        end    
    end
    
    return tp, fp, tn, fn, total_pos, total_neg
end

---------------------MAIN-RUN--------------------------------------
--[[
-- creating or loading the log tensor that stores evaluations
if opt.epochID == 1 then
  mylog = torch.Tensor(opt.epochs+1,20):zero()
  prlog = {}
else
  mylog = torch.load( opt.dataDir..'Results/'..saveName..'.t7', mylog )
end

epoch = opt.epochID

for counter=opt.epochID,opt.epochs do
   
  train()
   
  if opt.less_eval then
  if counter % 10 == 0 then
    evaluateAll( epoch )
  end
  else
    evaluateAll( epoch )
  end
    
  paths.mkdir(opt.dataDir..'Result/')
  paths.mkdir(opt.dataDir..'Model/')
  torch.save( opt.dataDir..'Result/'..saveName..'.t7', mylog )
  torch.save( opt.dataDir..'Result/pr_'..saveName..'.t7', prlog )
  
  if opt.saveModel then
    torch.save( opt.dataDir..'/Model/'..saveName..'.t7', model )
  end 
end
]]
--Perform single train/test
model = build_model()
ConvInit('cudnn.SpatialConvolution')
ConvInit('nn.SpatialConvolution')
BNInit('fbnn.SpatialBatchNormalization')
for k,v in pairs(model:findModules('nn.Linear')) do
   v.bias:zero()
end
 criterion = nn.BCECriterion()
model:cuda()
criterion:cuda()

print('################## TRAINING ##################')
for i=1, opt.epochs do
    train()
end

make_prediction(valData[1], test_feature)

torch.save( opt.dataDir..'Result/predictions-'..saveName..'.t7', predictions)
  
pfile = io.open(opt.dataDir..'Result/prediction-'..saveName..'.txt', 'w')
for p=1, #predictions do
    pfile:write(predictions[p]..'\n')
end
pfile:close()

--evaluate performance (considers classification threshold at 0.5)
test_ppiFile = Csv(opt.test..'.csv', "r")
test_ppi = test_ppiFile:readall()
test_ppiFile:close()

pred_ppiFile = Csv(opt.dataDir..'Result/prediction-'..saveName..'.txt', "r")
pred_ppi = pred_ppiFile:readall()
pred_ppiFile:close()

tp, fp, tn, fn, total_pos, total_neg = get_performance(test_ppi, pred_ppi)
print('\ntp = '..tp..'\ntn = '..tn..'\nfp = '..fp..'\nfn = '..fn..'\n')

accuracy = tp / #test_ppi
precision = tp / (tp + fp)
recall = tp / total_pos
specificity = tn / (tn + fn)
f1 = 2. * precision * recall / (precision + recall)
mcc = (tp * tn - fp * fn) / (((tp + tn) * (tp + fn) * (fp + tn) * (fp + fn)) ^ 0.5)
print('accuracy = '.. accuracy, 
    '\nprecision = '..precision, 
    '\nrecall = '..recall, 
    '\nspecificity = '..specificity, 
    '\nf1 = '..f1, 
    '\nmcc = '..mcc)
--write results to file
metrics = io.open(opt.dataDir..'Result/performance-'..saveName..'.txt', 'w')
metrics:write('accuracy = '.. accuracy, '\nprecision = '..precision, '\nrecall = '..recall, '\nspecificity = '..specificity, '\nf1 = '..f1, '\nmcc = '..mcc..'\n')
metrics:write('\ntp = '..tp..'\ntn = '..tn..'\nfp = '..fp..'\nfn = '..fn..'\n')
metrics:close()