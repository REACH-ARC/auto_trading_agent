//+------------------------------------------------------------------+
//|  AI_Analyst.mq5                                                  |
//|  MT5 Expert Advisor — Real-Time AI Trading Analyst               |
//|                                                                  |
//|  Uses MT5 built-in WebRequest — NO external libraries needed.    |
//|  Before attaching: Tools → Options → Expert Advisors →           |
//|    "Allow WebRequests" and add  http://127.0.0.1:8000            |
//+------------------------------------------------------------------+
#property copyright "MT5 AI Analyst Bot"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//+------------------------------------------------------------------+
//|  Input parameters                                                |
//+------------------------------------------------------------------+
input group "=== Connection ==="
input string InpServerUrl   = "http://127.0.0.1:8000";  // Python backend URL
input int    InpTimeoutMs   = 30000;                    // HTTP timeout (ms)

input group "=== Data ==="
input int    InpBarsPerTF       = 100;   // OHLCV bars per timeframe
input bool   InpSendOnNewM15Bar = true;  // Send only on new M15 bar (recommended)

input group "=== Chart ==="
input bool   InpShowObjects = true;   // Draw SL/TP/entry objects
input bool   InpShowLabel   = true;   // Show signal info label
input int    InpArrowSize   = 3;      // Entry arrow size (1-5)
input int    InpLineWidth   = 2;      // SL/TP line width

input group "=== Trading ==="
input bool   InpAutoTrade    = false;    // Enable auto order placement
input double InpLotOverride  = 0.0;     // Lot size override (0 = use bot sizing)
input bool   InpUseSlippage  = true;    // Apply 3-pip slippage tolerance
input int    InpMagicNumber  = 202400;  // EA magic number

//+------------------------------------------------------------------+
//|  Signal structure                                                |
//+------------------------------------------------------------------+
struct SignalData
{
   string symbol;
   string direction;   // "BUY" | "SELL" | "NO_TRADE"
   double entry;
   double sl;
   double tp1;
   double tp2;
   double tp3;
   int    confidence;
   double lot_size;
   bool   approved;
   string reasoning;
   string invalidation;
   double risk_reward;
   long   signal_id;
   bool   valid;
};

//+------------------------------------------------------------------+
//|  Globals                                                         |
//+------------------------------------------------------------------+
CTrade   g_trade;
datetime g_last_m15_bar   = 0;
long     g_last_signal_id = -1;

static const string OBJ_PREFIX  = "AIA_";
static const string SIGNAL_URL  = "/signal";

//+------------------------------------------------------------------+
//|  OnInit                                                          |
//+------------------------------------------------------------------+
int OnInit()
{
   if(InpBarsPerTF < 10 || InpBarsPerTF > 500)
   {
      Alert("InpBarsPerTF must be between 10 and 500");
      return INIT_PARAMETERS_INCORRECT;
   }

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpUseSlippage ? 30 : 0);

   Print("AI Analyst EA v2 ready — server: ", InpServerUrl);
   Print("NOTE: Enable WebRequests for ", InpServerUrl,
         " in Tools → Options → Expert Advisors");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//|  OnDeinit                                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   _DeleteObjects();
   Print("AI Analyst EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//|  OnTick — detect new M15 bar, send OHLCV, draw signal           |
//+------------------------------------------------------------------+
void OnTick()
{
   if(InpSendOnNewM15Bar)
   {
      datetime current_m15 = iTime(_Symbol, PERIOD_M15, 0);
      if(current_m15 == g_last_m15_bar) return;
      g_last_m15_bar = current_m15;
   }

   SignalData sig;
   if(_RequestSignal(sig))
   {
      if(sig.signal_id == g_last_signal_id)
      {
         Print("Duplicate signal id=", sig.signal_id, " — skipped");
         return;
      }
      g_last_signal_id = sig.signal_id;

      Print(StringFormat(
         "Signal #%d | %s %s | conf=%d%% | entry=%.5f SL=%.5f TP1=%.5f | approved=%s",
         sig.signal_id, sig.symbol, sig.direction,
         sig.confidence, sig.entry, sig.sl, sig.tp1,
         sig.approved ? "YES" : "NO"
      ));

      if(InpShowObjects || InpShowLabel)
         _DrawSignal(sig);

      if(InpAutoTrade && sig.approved && sig.direction != "NO_TRADE")
         _PlaceOrder(sig);
   }
}

//+------------------------------------------------------------------+
//|  Build request body, call /signal, parse response               |
//+------------------------------------------------------------------+
bool _RequestSignal(SignalData &sig)
{
   string body = _BuildRequestJson();
   if(body == "") return false;

   uchar  post_data[];
   uchar  response[];
   string headers = "Content-Type: application/json\r\n";
   string resp_headers;

   StringToCharArray(body, post_data, 0, StringLen(body));

   string url = InpServerUrl + SIGNAL_URL;
   int res = WebRequest("POST", url, headers, InpTimeoutMs,
                        post_data, response, resp_headers);

   if(res < 0)
   {
      int err = GetLastError();
      if(err == 4014)
         Print("ERROR 4014: WebRequest not allowed. Enable it in Tools → Options → Expert Advisors for ", InpServerUrl);
      else
         Print("WebRequest failed. Error=", err, "  URL=", url);
      return false;
   }

   if(res != 200)
   {
      Print("Server returned HTTP ", res, " — check Python server logs");
      return false;
   }

   string json = CharArrayToString(response);
   return _ParseSignalJson(json, sig);
}

//+------------------------------------------------------------------+
//|  Build the JSON body for POST /signal                            |
//+------------------------------------------------------------------+
string _BuildRequestJson()
{
   double equity      = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   int    open_trades = PositionsTotal();
   double daily_pnl   = equity - balance;

   string m15 = _BuildBarsJson(PERIOD_M15);
   string h1  = _BuildBarsJson(PERIOD_H1);
   string h4  = _BuildBarsJson(PERIOD_H4);
   string d1  = _BuildBarsJson(PERIOD_D1);

   if(m15 == "" && h1 == "" && h4 == "" && d1 == "")
   {
      Print("ERROR: No bar data available");
      return "";
   }

   return StringFormat(
      "{\"symbol\":\"%s\","
      "\"equity\":%.2f,\"balance\":%.2f,"
      "\"open_trades\":%d,\"daily_pnl\":%.2f,"
      "\"bars_m15\":[%s],"
      "\"bars_h1\":[%s],"
      "\"bars_h4\":[%s],"
      "\"bars_d1\":[%s]}",
      _Symbol,
      equity, balance,
      open_trades, daily_pnl,
      m15, h1, h4, d1
   );
}

string _BuildBarsJson(const ENUM_TIMEFRAMES period)
{
   int count = MathMin(InpBarsPerTF, iBars(_Symbol, period));
   if(count < 5) return "";

   string arr = "";
   for(int i = count - 1; i >= 0; i--)
   {
      datetime t   = iTime  (_Symbol, period, i);
      double   op  = iOpen  (_Symbol, period, i);
      double   hi  = iHigh  (_Symbol, period, i);
      double   lo  = iLow   (_Symbol, period, i);
      double   cl  = iClose (_Symbol, period, i);
      long     vol = iVolume(_Symbol, period, i);

      // ISO-8601 timestamp
      string ts = TimeToString(t, TIME_DATE | TIME_SECONDS);
      StringReplace(ts, ".", "-");
      StringReplace(ts, " ", "T");
      ts += "+00:00";

      if(arr != "") arr += ",";
      arr += StringFormat(
         "{\"time\":\"%s\",\"open\":%s,\"high\":%s,\"low\":%s,\"close\":%s,\"volume\":%d}",
         ts,
         DoubleToString(op, _Digits), DoubleToString(hi, _Digits),
         DoubleToString(lo, _Digits), DoubleToString(cl, _Digits),
         (int)vol
      );
   }
   return arr;
}

//+------------------------------------------------------------------+
//|  Parse /signal JSON response into SignalData                     |
//+------------------------------------------------------------------+
string _JsonStr(const string json, const string key)
{
   string search = "\"" + key + "\":\"";
   int start = StringFind(json, search);
   if(start < 0) return "";
   start += StringLen(search);
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
}

double _JsonNum(const string json, const string key)
{
   string search = "\"" + key + "\":";
   int start = StringFind(json, search);
   if(start < 0) return 0.0;
   start += StringLen(search);
   string val = "";
   for(int i = start; i < start + 30 && i < StringLen(json); i++)
   {
      ushort c = StringGetCharacter(json, i);
      if(c == ',' || c == '}' || c == ' ' || c == '\n' || c == ']') break;
      val += ShortToString(c);
   }
   if(val == "null" || val == "") return 0.0;
   return StringToDouble(val);
}

bool _JsonBool(const string json, const string key)
{
   string search = "\"" + key + "\":";
   int start = StringFind(json, search);
   if(start < 0) return false;
   start += StringLen(search);
   return StringFind(StringSubstr(json, start, 5), "true") >= 0;
}

bool _ParseSignalJson(const string json, SignalData &sig)
{
   sig.valid      = false;
   sig.symbol     = _JsonStr(json, "symbol");
   sig.direction  = _JsonStr(json, "direction");
   sig.reasoning  = _JsonStr(json, "reasoning");
   sig.invalidation = _JsonStr(json, "invalidation");
   sig.entry      = _JsonNum(json, "entry");
   sig.sl         = _JsonNum(json, "sl");
   sig.tp1        = _JsonNum(json, "tp1");
   sig.tp2        = _JsonNum(json, "tp2");
   sig.tp3        = _JsonNum(json, "tp3");
   sig.confidence = (int)_JsonNum(json, "confidence");
   sig.lot_size   = _JsonNum(json, "lot_size");
   sig.risk_reward= _JsonNum(json, "risk_reward");
   sig.signal_id  = (long)_JsonNum(json, "signal_id");
   sig.approved   = _JsonBool(json, "approved");

   if(sig.symbol == "" || sig.direction == "") return false;
   if(sig.direction != "BUY" && sig.direction != "SELL" && sig.direction != "NO_TRADE")
      return false;

   sig.valid = true;
   return true;
}

//+------------------------------------------------------------------+
//|  Draw signal objects on chart                                    |
//+------------------------------------------------------------------+
void _DrawSignal(const SignalData &sig)
{
   _DeleteObjects();
   if(sig.direction == "NO_TRADE") return;

   datetime now  = TimeCurrent();
   bool     is_buy = sig.direction == "BUY";

   if(InpShowObjects)
   {
      // Entry arrow
      string arrow = OBJ_PREFIX + "Arrow";
      ObjectCreate(0, arrow, is_buy ? OBJ_ARROW_BUY : OBJ_ARROW_SELL, 0, now, sig.entry);
      ObjectSetInteger(0, arrow, OBJPROP_COLOR, is_buy ? clrDodgerBlue : clrOrangeRed);
      ObjectSetInteger(0, arrow, OBJPROP_WIDTH, InpArrowSize);
      ObjectSetInteger(0, arrow, OBJPROP_ANCHOR, is_buy ? ANCHOR_TOP : ANCHOR_BOTTOM);
      ObjectSetString (0, arrow, OBJPROP_TOOLTIP,
                       StringFormat("%s Entry @ %.5f", sig.direction, sig.entry));

      // SL line — red dashed
      if(sig.sl > 0) _DrawHLine("SL",  sig.sl, clrRed,         STYLE_DASH,
                                 StringFormat("SL @ %.5f", sig.sl));

      // TP lines — green shades
      _DrawHLine("TP1", sig.tp1, clrLimeGreen,      STYLE_DASH,
                  StringFormat("TP1 @ %.5f  (1.5R)", sig.tp1));
      _DrawHLine("TP2", sig.tp2, clrMediumSeaGreen, STYLE_DASH,
                  StringFormat("TP2 @ %.5f  (2.5R)", sig.tp2));
      _DrawHLine("TP3", sig.tp3, clrDarkGreen,      STYLE_DASH,
                  StringFormat("TP3 @ %.5f  (4.0R)", sig.tp3));
   }

   if(InpShowLabel)
      _DrawLabel(sig);

   ChartRedraw(0);
}

void _DrawHLine(const string tag, const double price,
                const color clr, const ENUM_LINE_STYLE style,
                const string tooltip)
{
   if(price <= 0) return;
   string name = OBJ_PREFIX + tag;
   ObjectCreate(0, name, OBJ_HLINE, 0, 0, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR,   clr);
   ObjectSetInteger(0, name, OBJPROP_STYLE,   style);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,   InpLineWidth);
   ObjectSetString (0, name, OBJPROP_TOOLTIP, tooltip);
}

void _DrawLabel(const SignalData &sig)
{
   string reason = sig.reasoning;
   if(StringLen(reason) > 80)
      reason = StringSubstr(reason, 0, 80) + "...";

   string rr = sig.risk_reward > 0 ? StringFormat("%.2f", sig.risk_reward) : "n/a";

   string text = StringFormat(
      "AI Signal #%d\n"
      "%s  %s  Conf: %d%%\n"
      "Entry: %.5f   SL: %.5f\n"
      "TP1: %.5f   TP2: %.5f   TP3: %.5f\n"
      "RR: %s   Lot: %.2f   Approved: %s\n"
      "%s",
      sig.signal_id,
      sig.symbol, sig.direction, sig.confidence,
      sig.entry, sig.sl,
      sig.tp1, sig.tp2, sig.tp3,
      rr, sig.lot_size, sig.approved ? "YES" : "NO",
      reason
   );

   string name = OBJ_PREFIX + "Label";
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 12);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 30);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,  9);
   ObjectSetString (0, name, OBJPROP_FONT,      "Consolas");
   ObjectSetString (0, name, OBJPROP_TEXT,      text);
   ObjectSetInteger(0, name, OBJPROP_COLOR,
                    sig.direction == "BUY" ? clrDodgerBlue : clrOrangeRed);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
}

void _DeleteObjects()
{
   string tags[] = {"Arrow","SL","TP1","TP2","TP3","Label"};
   for(int i = 0; i < ArraySize(tags); i++)
      ObjectDelete(0, OBJ_PREFIX + tags[i]);
}

//+------------------------------------------------------------------+
//|  Auto-trade: place market order with SL and TP1                 |
//+------------------------------------------------------------------+
void _PlaceOrder(const SignalData &sig)
{
   if(sig.entry <= 0 || sig.sl <= 0)
   {
      Print("Auto-trade skipped: missing entry or SL");
      return;
   }

   double lot = InpLotOverride > 0 ? InpLotOverride : sig.lot_size;
   if(lot <= 0) { Print("Auto-trade skipped: lot size is 0"); return; }

   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lot = MathMax(vmin, MathMin(vmax, MathRound(lot / step) * step));

   bool   is_buy = sig.direction == "BUY";
   double price  = is_buy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                          : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl     = NormalizeDouble(sig.sl,  _Digits);
   double tp     = NormalizeDouble(sig.tp1, _Digits);
   string comment = StringFormat("AIA#%d conf=%d%%", sig.signal_id, sig.confidence);

   bool ok = is_buy ? g_trade.Buy (lot, _Symbol, price, sl, tp, comment)
                    : g_trade.Sell(lot, _Symbol, price, sl, tp, comment);

   if(ok)
      Print(StringFormat("Order placed: %s %.2f lots @ %.5f  SL=%.5f  TP=%.5f",
            sig.direction, lot, price, sl, tp));
   else
      Print(StringFormat("Order FAILED: retcode=%d  %s",
            g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription()));
}
//+------------------------------------------------------------------+
