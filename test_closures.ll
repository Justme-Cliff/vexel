; ModuleID = "vexel_module"
target triple = "x86_64-pc-windows-msvc"
target datalayout = ""

%"vx_array" = type {i8*, i64, i64}
%"vx_dict" = type {i8*, i8*, i64, i64}
declare i32 @"printf"(i8* %".1", ...)

declare i8* @"malloc"(i64 %".1")

declare void @"free"(i8* %".1")

declare i64 @"strlen"(i8* %".1")

declare i8* @"memcpy"(i8* %".1", i8* %".2", i64 %".3")

declare i32 @"sprintf"(i8* %".1", ...)

declare double @"sqrt"(double %".1")

declare double @"fabs"(double %".1")

declare i64 @"llabs"(i64 %".1")

declare double @"pow"(double %".1", double %".2")

declare double @"floor"(double %".1")

declare double @"ceil"(double %".1")

declare i32 @"strcmp"(i8* %".1", i8* %".2")

declare i8* @"realloc"(i8* %".1", i64 %".2")

declare i32 @"strncmp"(i8* %".1", i8* %".2", i64 %".3")

declare i8* @"strstr"(i8* %".1", i8* %".2")

declare i32 @"toupper"(i32 %".1")

declare i32 @"tolower"(i32 %".1")

declare void @"exit"(i32 %".1")

declare i32 @"rand"()

declare void @"srand"(i32 %".1")

declare i64 @"time"(i8* %".1")

declare double @"sin"(double %".1")

declare double @"cos"(double %".1")

declare double @"tan"(double %".1")

declare double @"log"(double %".1")

declare double @"log2"(double %".1")

declare i8* @"fopen"(i8* %".1", i8* %".2")

declare i32 @"fclose"(i8* %".1")

declare i64 @"fread"(i8* %".1", i64 %".2", i64 %".3", i8* %".4")

declare i64 @"fwrite"(i8* %".1", i64 %".2", i64 %".3", i8* %".4")

declare i32 @"fseek"(i8* %".1", i64 %".2", i32 %".3")

declare i64 @"ftell"(i8* %".1")

declare double @"round"(double %".1")

declare double @"atan2"(double %".1", double %".2")

declare i8* @"_getcwd"(i8* %".1", i32 %".2")

declare i32 @"_mkdir"(i8* %".1")

declare i32 @"remove"(i8* %".1")

declare i32 @"_rmdir"(i8* %".1")

declare i8* @"strerror"(i32 %".1")

declare i64 @"atoll"(i8* %".1")

declare double @"atof"(i8* %".1")

declare i64 @"strftime"(i8* %".1", i64 %".2", i8* %".3", i8* %".4")

declare i8* @"localtime"(i8* %".1")

declare i8* @"fgets"(i8* %".1", i32 %".2", i8* %".3")

@"__vx_error_buf" = private global [512 x i8] c"\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
define %"vx_array"* @"apply"(%"vx_array"* %"arr", i8* %"f")
{
entry:
  %"arr.1" = alloca %"vx_array"*
  store %"vx_array"* %"arr", %"vx_array"** %"arr.1"
  %"f.1" = alloca i8*
  store i8* %"f", i8** %"f.1"
  %".6" = call i8* @"malloc"(i64 8)
  %".7" = bitcast i8* %".6" to i64*
  %".8" = call i8* @"malloc"(i64 24)
  %".9" = bitcast i8* %".8" to %"vx_array"*
  %".10" = getelementptr inbounds %"vx_array", %"vx_array"* %".9", i32 0, i32 0
  store i8* %".6", i8** %".10"
  %".12" = getelementptr inbounds %"vx_array", %"vx_array"* %".9", i32 0, i32 1
  store i64 0, i64* %".12"
  %".14" = getelementptr inbounds %"vx_array", %"vx_array"* %".9", i32 0, i32 2
  store i64 4, i64* %".14"
  %"result" = alloca %"vx_array"*
  store %"vx_array"* %".9", %"vx_array"** %"result"
  %"arr.2" = load %"vx_array"*, %"vx_array"** %"arr.1"
  %".17" = getelementptr inbounds %"vx_array", %"vx_array"* %"arr.2", i32 0, i32 1
  %".18" = load i64, i64* %".17"
  %"_fe_i" = alloca i64
  store i64 0, i64* %"_fe_i"
  %"x" = alloca i64
  br label %"fe.check"
fe.check:
  %".21" = load i64, i64* %"_fe_i"
  %".22" = icmp slt i64 %".21", %".18"
  br i1 %".22", label %"fe.body", label %"fe.exit"
fe.body:
  %".24" = getelementptr inbounds %"vx_array", %"vx_array"* %"arr.2", i32 0, i32 0
  %".25" = load i8*, i8** %".24"
  %".26" = bitcast i8* %".25" to i64*
  %".27" = getelementptr inbounds i64, i64* %".26", i64 %".21"
  %".28" = load i64, i64* %".27"
  store i64 %".28", i64* %"x"
  %"result.1" = load %"vx_array"*, %"vx_array"** %"result"
  %".30" = load i8*, i8** %"f.1"
  %".31" = bitcast i8* %".30" to i64 (i64)*
  %"x.1" = load i64, i64* %"x"
  %".32" = call i64 %".31"(i64 %"x.1")
  %".33" = alloca i64
  store i64 %".32", i64* %".33"
  %".35" = bitcast i64* %".33" to i8*
  %".36" = bitcast %"vx_array"* %"result.1" to i8*
  call void @"__vx_array_push"(i8* %".36", i8* %".35", i64 8)
  %".38" = load i64, i64* %"_fe_i"
  %".39" = add i64 %".38", 1
  store i64 %".39", i64* %"_fe_i"
  br label %"fe.check"
fe.exit:
  %"result.2" = load %"vx_array"*, %"vx_array"** %"result"
  ret %"vx_array"* %"result.2"
}

define i64 @"sum"(%"vx_array"* %"nums")
{
entry:
  %"nums.1" = alloca %"vx_array"*
  store %"vx_array"* %"nums", %"vx_array"** %"nums.1"
  %"total" = alloca i64
  store i64 0, i64* %"total"
  %"nums.2" = load %"vx_array"*, %"vx_array"** %"nums.1"
  %".5" = getelementptr inbounds %"vx_array", %"vx_array"* %"nums.2", i32 0, i32 1
  %".6" = load i64, i64* %".5"
  %"_fe_i" = alloca i64
  store i64 0, i64* %"_fe_i"
  %"n" = alloca i64
  br label %"fe.check"
fe.check:
  %".9" = load i64, i64* %"_fe_i"
  %".10" = icmp slt i64 %".9", %".6"
  br i1 %".10", label %"fe.body", label %"fe.exit"
fe.body:
  %".12" = getelementptr inbounds %"vx_array", %"vx_array"* %"nums.2", i32 0, i32 0
  %".13" = load i8*, i8** %".12"
  %".14" = bitcast i8* %".13" to i64*
  %".15" = getelementptr inbounds i64, i64* %".14", i64 %".9"
  %".16" = load i64, i64* %".15"
  store i64 %".16", i64* %"n"
  %"total.1" = load i64, i64* %"total"
  %"n.1" = load i64, i64* %"n"
  %".18" = add i64 %"total.1", %"n.1"
  store i64 %".18", i64* %"total"
  %".20" = load i64, i64* %"_fe_i"
  %".21" = add i64 %".20", 1
  store i64 %".21", i64* %"_fe_i"
  br label %"fe.check"
fe.exit:
  %"total.2" = load i64, i64* %"total"
  ret i64 %"total.2"
}

define i32 @"main"()
{
entry:
  %".2" = bitcast i64 (i64)* @"__lambda_0" to i8*
  %"double" = alloca i8*
  store i8* %".2", i8** %"double"
  %".4" = mul i64 5, 8
  %".5" = call i8* @"malloc"(i64 %".4")
  %".6" = bitcast i8* %".5" to i64*
  %".7" = getelementptr inbounds i64, i64* %".6", i64 0
  store i64 1, i64* %".7"
  %".9" = getelementptr inbounds i64, i64* %".6", i64 1
  store i64 2, i64* %".9"
  %".11" = getelementptr inbounds i64, i64* %".6", i64 2
  store i64 3, i64* %".11"
  %".13" = getelementptr inbounds i64, i64* %".6", i64 3
  store i64 4, i64* %".13"
  %".15" = getelementptr inbounds i64, i64* %".6", i64 4
  store i64 5, i64* %".15"
  %".17" = call i8* @"malloc"(i64 24)
  %".18" = bitcast i8* %".17" to %"vx_array"*
  %".19" = getelementptr inbounds %"vx_array", %"vx_array"* %".18", i32 0, i32 0
  store i8* %".5", i8** %".19"
  %".21" = getelementptr inbounds %"vx_array", %"vx_array"* %".18", i32 0, i32 1
  store i64 5, i64* %".21"
  %".23" = getelementptr inbounds %"vx_array", %"vx_array"* %".18", i32 0, i32 2
  store i64 5, i64* %".23"
  %"nums" = alloca %"vx_array"*
  store %"vx_array"* %".18", %"vx_array"** %"nums"
  %"nums.1" = load %"vx_array"*, %"vx_array"** %"nums"
  %"double.1" = load i8*, i8** %"double"
  %".26" = call %"vx_array"* @"apply"(%"vx_array"* %"nums.1", i8* %"double.1")
  %"doubled" = alloca %"vx_array"*
  store %"vx_array"* %".26", %"vx_array"** %"doubled"
  %"doubled.1" = load %"vx_array"*, %"vx_array"** %"doubled"
  %".28" = getelementptr inbounds %"vx_array", %"vx_array"* %"doubled.1", i32 0, i32 1
  %".29" = load i64, i64* %".28"
  %"_fe_i" = alloca i64
  store i64 0, i64* %"_fe_i"
  %"n" = alloca i64
  br label %"fe.check"
fe.check:
  %".32" = load i64, i64* %"_fe_i"
  %".33" = icmp slt i64 %".32", %".29"
  br i1 %".33", label %"fe.body", label %"fe.exit"
fe.body:
  %".35" = getelementptr inbounds %"vx_array", %"vx_array"* %"doubled.1", i32 0, i32 0
  %".36" = load i8*, i8** %".35"
  %".37" = bitcast i8* %".36" to i64*
  %".38" = getelementptr inbounds i64, i64* %".37", i64 %".32"
  %".39" = load i64, i64* %".38"
  store i64 %".39", i64* %"n"
  %"n.1" = load i64, i64* %"n"
  %".41" = getelementptr inbounds [5 x i8], [5 x i8]* @".str.0", i32 0, i32 0
  %".42" = call i32 (i8*, ...) @"printf"(i8* %".41", i64 %"n.1")
  %".43" = getelementptr inbounds [2 x i8], [2 x i8]* @".str.1", i32 0, i32 0
  %".44" = call i32 (i8*, ...) @"printf"(i8* %".43")
  %".45" = load i64, i64* %"_fe_i"
  %".46" = add i64 %".45", 1
  store i64 %".46", i64* %"_fe_i"
  br label %"fe.check"
fe.exit:
  %".49" = mul i64 5, 8
  %".50" = call i8* @"malloc"(i64 %".49")
  %".51" = bitcast i8* %".50" to i64*
  %".52" = getelementptr inbounds i64, i64* %".51", i64 0
  store i64 1, i64* %".52"
  %".54" = getelementptr inbounds i64, i64* %".51", i64 1
  store i64 2, i64* %".54"
  %".56" = getelementptr inbounds i64, i64* %".51", i64 2
  store i64 3, i64* %".56"
  %".58" = getelementptr inbounds i64, i64* %".51", i64 3
  store i64 4, i64* %".58"
  %".60" = getelementptr inbounds i64, i64* %".51", i64 4
  store i64 5, i64* %".60"
  %".62" = call i8* @"malloc"(i64 24)
  %".63" = bitcast i8* %".62" to %"vx_array"*
  %".64" = getelementptr inbounds %"vx_array", %"vx_array"* %".63", i32 0, i32 0
  store i8* %".50", i8** %".64"
  %".66" = getelementptr inbounds %"vx_array", %"vx_array"* %".63", i32 0, i32 1
  store i64 5, i64* %".66"
  %".68" = getelementptr inbounds %"vx_array", %"vx_array"* %".63", i32 0, i32 2
  store i64 5, i64* %".68"
  %".70" = call i64 @"sum"(%"vx_array"* %".63")
  %".71" = getelementptr inbounds [5 x i8], [5 x i8]* @".str.0", i32 0, i32 0
  %".72" = call i32 (i8*, ...) @"printf"(i8* %".71", i64 %".70")
  %".73" = getelementptr inbounds [2 x i8], [2 x i8]* @".str.1", i32 0, i32 0
  %".74" = call i32 (i8*, ...) @"printf"(i8* %".73")
  %".75" = mul i64 2, 8
  %".76" = call i8* @"malloc"(i64 %".75")
  %".77" = bitcast i8* %".76" to i64*
  %".78" = getelementptr inbounds i64, i64* %".77", i64 0
  store i64 10, i64* %".78"
  %".80" = getelementptr inbounds i64, i64* %".77", i64 1
  store i64 20, i64* %".80"
  %".82" = call i8* @"malloc"(i64 24)
  %".83" = bitcast i8* %".82" to %"vx_array"*
  %".84" = getelementptr inbounds %"vx_array", %"vx_array"* %".83", i32 0, i32 0
  store i8* %".76", i8** %".84"
  %".86" = getelementptr inbounds %"vx_array", %"vx_array"* %".83", i32 0, i32 1
  store i64 2, i64* %".86"
  %".88" = getelementptr inbounds %"vx_array", %"vx_array"* %".83", i32 0, i32 2
  store i64 4, i64* %".88"
  %".90" = call i64 @"sum"(%"vx_array"* %".83")
  %".91" = getelementptr inbounds [5 x i8], [5 x i8]* @".str.0", i32 0, i32 0
  %".92" = call i32 (i8*, ...) @"printf"(i8* %".91", i64 %".90")
  %".93" = getelementptr inbounds [2 x i8], [2 x i8]* @".str.1", i32 0, i32 0
  %".94" = call i32 (i8*, ...) @"printf"(i8* %".93")
  ret i32 0
}

define private void @"__vx_array_push"(i8* %".1", i8* %".2", i64 %".3")
{
entry:
  %".5" = bitcast i8* %".1" to %"vx_array"*
  %".6" = getelementptr inbounds %"vx_array", %"vx_array"* %".5", i32 0, i32 1
  %".7" = getelementptr inbounds %"vx_array", %"vx_array"* %".5", i32 0, i32 2
  %".8" = getelementptr inbounds %"vx_array", %"vx_array"* %".5", i32 0, i32 0
  %".9" = load i64, i64* %".6"
  %".10" = load i64, i64* %".7"
  %".11" = icmp sge i64 %".9", %".10"
  br i1 %".11", label %"push.grow", label %"push.store"
push.grow:
  %".13" = mul i64 %".10", 2
  %".14" = icmp slt i64 %".13", 4
  %".15" = select  i1 %".14", i64 4, i64 %".13"
  %".16" = mul i64 %".15", %".3"
  %".17" = load i8*, i8** %".8"
  %".18" = call i8* @"realloc"(i8* %".17", i64 %".16")
  store i8* %".18", i8** %".8"
  store i64 %".15", i64* %".7"
  br label %"push.store"
push.store:
  %".22" = load i8*, i8** %".8"
  %".23" = mul i64 %".9", %".3"
  %".24" = getelementptr i8, i8* %".22", i64 %".23"
  %".25" = call i8* @"memcpy"(i8* %".24", i8* %".2", i64 %".3")
  %".26" = add i64 %".9", 1
  store i64 %".26", i64* %".6"
  ret void
}

define i64 @"__lambda_0"(i64 %"x")
{
entry:
  %"x.1" = alloca i64
  store i64 %"x", i64* %"x.1"
  %"x.2" = load i64, i64* %"x.1"
  %".4" = mul i64 %"x.2", 2
  ret i64 %".4"
}

@".str.0" = private constant [5 x i8] c"%lld\00"
@".str.1" = private constant [2 x i8] c"\0a\00"
