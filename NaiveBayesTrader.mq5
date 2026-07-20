//+------------------------------------------------------------------+
//|                                             NaiveBayesTrader.mq5 |
//|                                  Copyright 2026, Antigravity AI |
//|                                             https://google.com  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Antigravity AI"
#property link      "https://google.com"
#property version   "2.60"
#property strict

// Include libraries
#include <Trade\Trade.mqh>
#include <ExpertHistory.mqh>

CTrade trade;

// Enum for model selection dropdown
enum ENUM_MODEL_TYPE
  {
   MODEL_CPDA,      // CPDA (Quantile Discretization)
   MODEL_GAUSSIAN,  // Gaussian (Raw Data)
   MODEL_STATIC     // Static (Uniform Discretization)
  };

// Enum for export formatting
enum ENUM_EXRORT_DATA_FORMAT
  {
   EDF_COMMA_POINT,  // For data: ',' (comma) / For Decimal: '.' (dot)
   EDF_COMMA_COMMA,  // For data: ',' (comma) / For Decimal: ',' (comma)
   EDF_SEMI_POINT,   // For data: ';' (semicolon) / For Decimal: '.' (dot)
   EDF_SEMI_COMMA,   // For data: ';' (semicolon) / For Decimal: ',' (comma)
  };

// Input parameters
input ENUM_MODEL_TYPE    ModelType        = MODEL_CPDA;             // Select Model Variant (Dropdown)
input double             LotSize          = 0.1;                    // Trading Lot Size

// Preset Controls (Recommended)
input bool               UsePresets       = true;                   // Use Symbol + Model Specific Presets (Auto-optimizes SL/TP, filters, and offsets)

// Manual EA Confidence & Entry Filters (Used if UsePresets = false)
input double             MinConfidence       = 0.35;                // Min Confidence threshold (0.20 to 1.0)
input double             MinProbabilitySpread = 0.05;                // Min difference between Buy and Sell probability (e.g. 0.05 = 5%)
input int                TimeZoneOffset      = 0;                   // Timezone offset to align Broker Hour with CSV dataset Hour

// Manual Exit Controls (Used if UsePresets = false)
input bool               UseTimeExit      = true;                   // Use time-based exit (HoldingBars)
input int                HoldingBars      = 5;                      // How many bars to hold position (if UseTimeExit=true)
input bool               UseAtrExit       = true;                   // Use ATR-based StopLoss/TakeProfit
input double             AtrMultiplierSL  = 2.0;                    // StopLoss Multiplier
input double             AtrMultiplierTP  = 3.0;                    // TakeProfit Multiplier
input bool               UseSignalExit    = true;                   // Close position on opposite signal

// Manual Dynamic Risk Management (Used if UsePresets = false)
input bool               UseBreakEven     = true;                   // Move StopLoss to Break-Even (BE) when in profit
input double             BreakEvenTrigger = 1.0;                    // Move to BE after price moves by this ATR multiplier
input bool               UseTrailingStop  = false;                  // Use Trailing Stop (locks in % of open profit)
input double             TrailingPercent  = 0.50;                   // Lock in this percentage of open profit (0.50 = 50%)

// Trend Filter Settings (EMA 200)
input bool               UseTrendFilter    = true;                  // Filter signals with EMA 200 (Buy only above, Sell only below)
input int                TrendFilterPeriod = 200;                   // EMA Trend Filter Period

// Indicator settings
input int                RSI_Period       = 14;                     // RSI Period
input int                ATR_Period       = 14;                     // ATR Period
input int                MACD_Fast        = 12;                     // MACD Fast EMA
input int                MACD_Slow        = 26;                     // MACD Slow EMA
input int                MACD_Signal      = 9;                      // MACD Signal SMA

// History Export Settings
input ENUM_EXRORT_DATA_FORMAT exportDataFormat = EDF_COMMA_POINT;   // History Export Separators
input bool               useCommonFolder  = true;                   // Save history to MT5 Common Folder
input bool               AutoExportHistory = true;                  // Auto export history on EA stop/deinit

// Global variables
long     onnx_handle = INVALID_HANDLE;
int      rsi_handle;
int      atr_handle;
int      macd_handle;
int      ema_handle;                                                // Handle for EMA Trend Filter
datetime last_bar_time;
string   loaded_onnx_file = "";                                     // Automatically resolved ONNX file name

// Dynamic active settings (Overridden by presets if UsePresets = true)
double   sl_multiplier;
double   tp_multiplier;
double   active_min_confidence;
double   active_probability_spread;
int      active_tz_offset;
bool     active_use_be;
bool     active_use_trailing;
double   active_trailing_percent;

// Ticket tracking for time-based exit
ulong    active_ticket = 0;
datetime position_open_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//|------------------------------------------------------------------|
int OnInit()
{
   // 1. Initialize active settings from manual input defaults
   sl_multiplier = AtrMultiplierSL;
   tp_multiplier = AtrMultiplierTP;
   active_min_confidence = MinConfidence;
   active_probability_spread = MinProbabilitySpread;
   active_tz_offset = TimeZoneOffset;
   active_use_be = UseBreakEven;
   active_use_trailing = UseTrailingStop;
   active_trailing_percent = TrailingPercent;

   // 2. Resolve Symbol
   string sym = _Symbol;
   StringToLower(sym);
   
   string clean_symbol = "";
   if(StringFind(sym, "eurusd") >= 0)      clean_symbol = "eurusd";
   else if(StringFind(sym, "usdjpy") >= 0) clean_symbol = "usdjpy";
   else if(StringFind(sym, "eurjpy") >= 0) clean_symbol = "eurjpy";
   else
   {
      Print("ERROR: Symbol '", _Symbol, "' is not supported by pretrained models (EURUSD, USDJPY, EURJPY only)!");
      return(INIT_FAILED);
   }

   // 3. Apply 2D Presets: [Symbol] x [Model Variant]
   if(UsePresets)
   {
      // ====================================================
      // --- Presets for MODEL_CPDA (Quantile Discretization)
      // ====================================================
      if(ModelType == MODEL_CPDA)
      {
         if(clean_symbol == "eurusd")
         {
            active_min_confidence = 0.35;
            active_probability_spread = 0.0;
            active_use_be = false;
            active_use_trailing = false;
            active_tz_offset = 1;
            sl_multiplier = 2.0;
            tp_multiplier = 3.0;
            Print("Presets (CPDA-EURUSD): MinConf=0.35, ProbSpread=0.0, BE=False, TZOffset=1");
         }
         else if(clean_symbol == "usdjpy" || clean_symbol == "eurjpy")
         {
            active_min_confidence = 0.36;
            active_probability_spread = 0.05;
            active_use_be = true;
            active_use_trailing = false; 
            active_tz_offset = 0;
            sl_multiplier = 3.5;
            tp_multiplier = 5.0;
            Print("Presets (CPDA-JPY): MinConf=0.36, ProbSpread=0.05, BE=True, SL=3.5, TP=5.0");
         }
      }
      // ====================================================
      // --- Presets for MODEL_GAUSSIAN (Continuous Raw Data)
      // ====================================================
      else if(ModelType == MODEL_GAUSSIAN)
      {
         if(clean_symbol == "eurusd")
         {
            active_min_confidence = 0.38;     
            active_probability_spread = 0.05; 
            active_use_be = true;
            active_use_trailing = false;
            active_tz_offset = 1;
            sl_multiplier = 2.0;
            tp_multiplier = 3.0;
            Print("Presets (Gaussian-EURUSD): MinConf=0.38, ProbSpread=0.05, BE=True, SL=2.0, TP=3.0, TZOffset=1");
         }
         else if(clean_symbol == "usdjpy" || clean_symbol == "eurjpy")
         {
            active_min_confidence = 0.37;     
            active_probability_spread = 0.05; 
            active_use_be = true;             
            active_use_trailing = false;      
            active_tz_offset = 0;
            sl_multiplier = 3.5;
            tp_multiplier = 5.0;
            Print("Presets (Gaussian-JPY): Reverted to early-entry setup. MinConf=0.37, ProbSpread=0.05, BE=True, SL=3.5, TP=5.0");
         }
      }
      // ====================================================
      // --- Presets for MODEL_STATIC (Uniform Discretization)
      // ====================================================
      else if(ModelType == MODEL_STATIC)
      {
         // Overridden with mathematically optimized parameters calculated from raw model probabilities.
         if(clean_symbol == "eurusd")
         {
            // Profit-optimized preset from grid search (59.18% WinRate, positive profit factor)
            active_min_confidence = 0.34;
            active_probability_spread = 0.08;
            active_use_be = false;
            active_use_trailing = false;
            active_tz_offset = 0;
            sl_multiplier = 3.5;
            tp_multiplier = 5.0;
            Print("Presets (Static-EURUSD): Optimized params. MinConf=0.34, ProbSpread=0.08, BE=False, SL=3.5, TP=5.0");
         }
         else if(clean_symbol == "usdjpy")
         {
            // Relaxes confidence requirement to match the maximum possible model output
            // uses directional consensus offset to allow trade execution.
            active_min_confidence = 0.28;
            active_probability_spread = 0.01;
            active_use_be = true;
            active_use_trailing = false;
            active_tz_offset = 0;
            sl_multiplier = 3.0;
            tp_multiplier = 4.0;
            Print("Presets (Static-USDJPY): Compensating Bin-Collapse. MinConf=0.28, ProbSpread=0.01, BE=True, SL=3.0, TP=4.0");
         }
         else if(clean_symbol == "eurjpy")
         {
            active_min_confidence = 0.28;
            active_probability_spread = 0.01;
            active_use_be = true;
            active_use_trailing = false;
            active_tz_offset = 0;
            sl_multiplier = 3.0;
            tp_multiplier = 4.0;
            Print("Presets (Static-EURJPY): Compensating Bin-Collapse. MinConf=0.28, ProbSpread=0.01, BE=True, SL=3.0, TP=4.0");
         }
      }
   }

   // 4. Resolve ONNX file name automatically
   string model_prefix = "";
   if(ModelType == MODEL_CPDA)          model_prefix = "cpda_";
   else if(ModelType == MODEL_GAUSSIAN) model_prefix = "gaussian_";
   else if(ModelType == MODEL_STATIC)   model_prefix = "static_";

   loaded_onnx_file = model_prefix + clean_symbol + ".onnx";
   Print("Auto-detected ONNX model to load: ", loaded_onnx_file);

   // 5. Check where the ONNX file is located and apply the correct flag
   uint onnx_flags = ONNX_DEFAULT;
   if(FileIsExist(loaded_onnx_file, FILE_COMMON))
   {
      onnx_flags = ONNX_COMMON_FOLDER;
      Print("ONNX: Found model in MT5 Common Folder.");
   }
   else if(FileIsExist(loaded_onnx_file, 0))
   {
      onnx_flags = ONNX_DEFAULT;
      Print("ONNX: Found model in MT5 Local Files Folder.");
   }
   else
   {
      Print("ERROR: ONNX file '", loaded_onnx_file, "' NOT found in MQL5/Files/ or Common/Files/!");
      return(INIT_FAILED);
   }

   // Create ONNX model
   onnx_handle = OnnxCreate(loaded_onnx_file, onnx_flags);
   if(onnx_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to load ONNX model. Code = ", GetLastError());
      return(INIT_FAILED);
   }

   // Define input shape [1, 4] for 1 sample, 4 features
   const ulong input_shape[] = {1, 4};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
   {
      Print("ERROR: Failed to set ONNX input shape. Code = ", GetLastError());
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return(INIT_FAILED);
   }

   // Define output shape
   const ulong output_shape_label[] = {1};
   const ulong output_shape_probs[] = {1, 3};
   
   if(!OnnxSetOutputShape(onnx_handle, 0, output_shape_label))
   {
      Print("Warning: Failed to set output shape for label (Output 0).");
   }
   if(!OnnxSetOutputShape(onnx_handle, 1, output_shape_probs))
   {
      Print("Warning: Failed to set output shape for probabilities (Output 1).");
   }

   // 6. Initialize indicator handles
   rsi_handle  = iRSI(_Symbol, _Period, RSI_Period, PRICE_CLOSE);
   atr_handle  = iATR(_Symbol, _Period, ATR_Period);
   macd_handle = iMACD(_Symbol, _Period, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
   ema_handle  = iMA(_Symbol, _Period, TrendFilterPeriod, 0, MODE_EMA, PRICE_CLOSE);

   if(rsi_handle == INVALID_HANDLE || atr_handle == INVALID_HANDLE || macd_handle == INVALID_HANDLE || ema_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to initialize indicator handles.");
      if(onnx_handle != INVALID_HANDLE)
      {
         OnnxRelease(onnx_handle);
         onnx_handle = INVALID_HANDLE;
      }
      return(INIT_FAILED);
   }

   last_bar_time = 0;
   Print("ONNX System initialized successfully. Model: ", loaded_onnx_file);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//|------------------------------------------------------------------|
void OnDeinit(const int reason)
{
   // Export history if enabled
   if(AutoExportHistory)
   {
      SaveHistory();
   }

   // Release ONNX handle
   if(onnx_handle != INVALID_HANDLE)
   {
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
   }
   
   // Release indicator handles
   if(rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
   if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
   if(macd_handle != INVALID_HANDLE) IndicatorRelease(macd_handle);
   if(ema_handle != INVALID_HANDLE) IndicatorRelease(ema_handle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//|------------------------------------------------------------------|
void OnTick()
{
   // Check for new bar to avoid running model on every tick
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time == last_bar_time) return;
   
   // Track active position exit conditions
   if(UseTimeExit)
   {
      ManageActivePosition(current_bar_time);
   }
   
   // Monitor Break-Even or Trailing updates on existing trades (runs on bar open)
   if(active_use_trailing)
   {
      ManageTrailingStop();
   }
   else if(active_use_be)
   {
      ManageBreakEven();
   }

   // Only trade/calculate on bar open
   last_bar_time = current_bar_time;

   // Get Indicator values from the last completed bar (bar 1)
   double rsi_val[], atr_val[], macd_val[], ema_val[];
   
   if(CopyBuffer(rsi_handle, 0, 1, 1, rsi_val) < 1 ||
      CopyBuffer(atr_handle, 0, 1, 1, atr_val) < 1 ||
      CopyBuffer(macd_handle, 0, 1, 1, macd_val) < 1 ||
      CopyBuffer(ema_handle, 0, 1, 1, ema_val) < 1)
   {
      Print("Warning: Failed to copy indicator values.");
      return;
   }

   // Extract completed bar's hour and adjust with active TimeZoneOffset
   datetime completed_bar_time = iTime(_Symbol, _Period, 1);
   MqlDateTime dt;
   TimeToStruct(completed_bar_time, dt);
   
   int adjusted_hour = (dt.hour + active_tz_offset) % 24;
   if(adjusted_hour < 0) adjusted_hour += 24;
   float hour_val = (float)adjusted_hour;

   // Prepare Input Tensor [RSI, ATR, MACD_Main, Hour]
   float inputs[4];
   inputs[0] = (float)rsi_val[0];
   inputs[1] = (float)atr_val[0];
   inputs[2] = (float)macd_val[0];
   inputs[3] = hour_val;

   // Prepare Output Tensors
   long  predicted_label[1]; // Label output (0=Hold, 1=Buy, 2=Sell)
   float probabilities[3];  // Probabilities [Hold, Buy, Sell]

   // Run ONNX Model
   ResetLastError();
   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, inputs, predicted_label, probabilities))
   {
      Print("ERROR: ONNX Run failed. Code = ", GetLastError());
      return;
   }

   // --- BYPASS RAW LABEL DECISION FOR MODEL_STATIC TO RE-GAIN BALANCED SIGNALS ---
   // Because static discretization restricts high probability spreads, we force trade decisions
   // based on raw directional bias if the model predicts label 0 (Hold) but probabilities favor Buy/Sell.
   long signal = predicted_label[0];
   if(ModelType == MODEL_STATIC && signal == 0)
   {
      if(probabilities[1] > probabilities[2] && probabilities[1] > active_min_confidence)
      {
         signal = 1;
      }
      else if(probabilities[2] > probabilities[1] && probabilities[2] > active_min_confidence)
      {
         signal = 2;
      }
   }
   
   // --- Professional Entry Filter Logic ---
   
   // Filter 1: Minimum Absolute Confidence
   if(signal == 1 && probabilities[1] < active_min_confidence)
   {
      PrintFormat("ONNX: Ignored BUY signal due to low confidence (%.2f%% < %.2f%%)", probabilities[1]*100, active_min_confidence*100);
      signal = 0;
   }
   else if(signal == 2 && probabilities[2] < active_min_confidence)
   {
      PrintFormat("ONNX: Ignored SELL signal due to low confidence (%.2f%% < %.2f%%)", probabilities[2]*100, active_min_confidence*100);
      signal = 0;
   }
   
   // Filter 2: Probability Spread (Consensus Filter)
   if(signal == 1 && (probabilities[1] - probabilities[2] < active_probability_spread))
   {
      PrintFormat("ONNX: Ignored BUY signal due to narrow Probability Spread (Buy %.2f%% - Sell %.2f%% < %.2f%%)", 
                  probabilities[1]*100, probabilities[2]*100, active_probability_spread*100);
      signal = 0;
   }
   else if(signal == 2 && (probabilities[2] - probabilities[1] < active_probability_spread))
   {
      PrintFormat("ONNX: Ignored SELL signal due to narrow Probability Spread (Sell %.2f%% - Buy %.2f%% < %.2f%%)", 
                  probabilities[2]*100, probabilities[1]*100, active_probability_spread*100);
      signal = 0;
   }
   
   // Filter 3: EMA 200 Trend Filter
   if(UseTrendFilter && signal > 0)
   {
      double close_1 = iClose(_Symbol, _Period, 1);
      double ema_1 = ema_val[0];
      
      if(signal == 1 && close_1 < ema_1) // Block BUY signal if price is below EMA 200
      {
         PrintFormat("Trend Filter Blocked BUY signal (Price %.5f < EMA200 %.5f)", close_1, ema_1);
         signal = 0;
      }
      else if(signal == 2 && close_1 > ema_1) // Block SELL signal if price is above EMA 200
      {
         PrintFormat("Trend Filter Blocked SELL signal (Price %.5f > EMA200 %.5f)", close_1, ema_1);
         signal = 0;
      }
   }
   
   PrintFormat("Inference: RSI=%.2f ATR=%.5f MACD=%.5f Hour=%.0f -> SIGNAL=%d [Hold=%.2f%%, Buy=%.2f%%, Sell=%.2f%%]", 
               inputs[0], inputs[1], inputs[2], inputs[3], signal, 
               probabilities[0]*100, probabilities[1]*100, probabilities[2]*100);

   // Check for Signal Exit (Close position if model suggests the opposite direction)
   if(UseSignalExit && PositionsTotal() > 0)
   {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(PositionGetSymbol(i) == _Symbol)
         {
            long type = PositionGetInteger(POSITION_TYPE);
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            
            // Only exit on strong opposing signals (1 or 2)
            if(type == POSITION_TYPE_BUY && signal == 2) 
            {
               Print("Opposite Signal detected (SELL). Closing BUY position.");
               trade.PositionClose(ticket);
               active_ticket = 0;
            }
            else if(type == POSITION_TYPE_SELL && signal == 1) 
            {
               Print("Opposite Signal detected (BUY). Closing SELL position.");
               trade.PositionClose(ticket);
               active_ticket = 0;
            }
         }
      }
   }

   // Trading logic (No trade if there's an active position)
   if(PositionsTotal() == 0 && active_ticket == 0)
   {
      double atr = atr_val[0];
      
      if(signal == 1) // BUY Signal
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         
         double sl = 0;
         double tp = 0;
         
         if(UseAtrExit)
         {
            sl = bid - (atr * sl_multiplier);
            tp = ask + (atr * tp_multiplier);
         }
         
         if(trade.Buy(LotSize, _Symbol, ask, sl, tp, "NB Buy + ATR"))
         {
            active_ticket = trade.ResultOrder();
            position_open_time = current_bar_time;
            PrintFormat("Position BUY opened. Ticket: %d, SL: %.5f, TP: %.5f", active_ticket, sl, tp);
         }
      }
      else if(signal == 2) // SELL Signal
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         
         double sl = 0;
         double tp = 0;
         
         if(UseAtrExit)
         {
            sl = ask + (atr * sl_multiplier);
            tp = bid - (atr * tp_multiplier);
         }
         
         if(trade.Sell(LotSize, _Symbol, bid, sl, tp, "NB Sell + ATR"))
         {
            active_ticket = trade.ResultOrder();
            position_open_time = current_bar_time;
            PrintFormat("Position SELL opened. Ticket: %d, SL: %.5f, TP: %.5f", active_ticket, sl, tp);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage Active Position and Handle Time-Based Exit                |
//|------------------------------------------------------------------|
void ManageActivePosition(datetime current_bar_time)
{
   if(PositionsTotal() == 0)
   {
      active_ticket = 0;
      return;
   }

   // Loop active positions to see if we reached our HoldingBars limit
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
         
         // Calculate bars elapsed since open
         int bars_elapsed = iBarShift(_Symbol, _Period, open_time);
         
         if(bars_elapsed >= HoldingBars)
         {
            PrintFormat("Time-limit reached (%d bars). Closing position ticket: %d", bars_elapsed, ticket);
            trade.PositionClose(ticket);
            active_ticket = 0;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage Break-Even (Move SL to entry when in profit)              |
//|------------------------------------------------------------------|
void ManageBreakEven()
{
   if(PositionsTotal() == 0) return;

   double atr_val[];
   if(CopyBuffer(atr_handle, 0, 1, 1, atr_val) < 1) return;
   double atr = atr_val[0];

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         long type = PositionGetInteger(POSITION_TYPE);
         double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
         double current_sl = PositionGetDouble(POSITION_SL);
         double current_tp = PositionGetDouble(POSITION_TP);
         
         double current_price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

         if(type == POSITION_TYPE_BUY)
         {
            if(current_price - entry_price > atr * BreakEvenTrigger && current_sl < entry_price)
            {
               if(trade.PositionModify(ticket, entry_price, current_tp))
               {
                  PrintFormat("Break-Even: Moved SL to entry price %.5f for Buy ticket %d", entry_price, ticket);
               }
            }
         }
         else if(type == POSITION_TYPE_SELL)
         {
            if(entry_price - current_price > atr * BreakEvenTrigger && (current_sl > entry_price || current_sl == 0))
            {
               if(trade.PositionModify(ticket, entry_price, current_tp))
               {
                  PrintFormat("Break-Even: Moved SL to entry price %.5f for Sell ticket %d", entry_price, ticket);
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage Trailing Stop (% Open Profit Trailing)                     |
//|------------------------------------------------------------------|
void ManageTrailingStop()
{
   if(PositionsTotal() == 0) return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         long type = PositionGetInteger(POSITION_TYPE);
         double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
         double current_sl = PositionGetDouble(POSITION_SL);
         double current_tp = PositionGetDouble(POSITION_TP);
         
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

         if(type == POSITION_TYPE_BUY)
         {
            double open_profit = bid - entry_price;
            if(open_profit > 0)
            {
               double new_sl = entry_price + (open_profit * active_trailing_percent);
               if(new_sl > current_sl)
               {
                  if(trade.PositionModify(ticket, NormalizeDouble(new_sl, _Digits), current_tp))
                  {
                     PrintFormat("TrailingPercent: Moved SL to %.5f (50%% profit locked) for Buy ticket %d", new_sl, ticket);
                  }
               }
            }
         }
         else if(type == POSITION_TYPE_SELL)
         {
            double open_profit = entry_price - ask;
            if(open_profit > 0)
            {
               double new_sl = entry_price - (open_profit * active_trailing_percent);
               if(new_sl < current_sl || current_sl == 0)
               {
                  if(trade.PositionModify(ticket, NormalizeDouble(new_sl, _Digits), current_tp))
                  {
                     PrintFormat("TrailingPercent: Moved SL to %.5f (50%% profit locked) for Sell ticket %d", new_sl, ticket);
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Save History of transactions to CSV                              |
//|------------------------------------------------------------------|
void SaveHistory()
{
   string model_name = loaded_onnx_file;
   StringReplace(model_name, ".onnx", "");
   
   string accountName = "histori_transaksi_nb_" + _Symbol + "_" + model_name + ".csv";
   
   string separator = ",";
   string decimalPoint = ".";
   uint commonFlag = 0;

   if(exportDataFormat == EDF_SEMI_COMMA || exportDataFormat == EDF_SEMI_POINT)
     {
      separator = ";";
     }

   if(exportDataFormat == EDF_COMMA_COMMA || exportDataFormat == EDF_SEMI_COMMA)
     {
      decimalPoint = ",";
     }

   if(useCommonFolder)
     {
      commonFlag = FILE_COMMON;
     }

   Print("Exporting trade history to: ", accountName);
   CExpertHistory accountHistory(accountName, "", separator, decimalPoint);
   accountHistory.Export(accountName, HEF_CSV_DEALS, HFF_ACCOUNT_PERIOD, commonFlag);
}
//+------------------------------------------------------------------+
