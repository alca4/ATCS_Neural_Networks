public class eek {
   public static int Duck(int a, int b)
   {
      if (b == 0) return a;
      return Duck(b, a % b);
   }

   public static int AnotherDuck(int a, int b)
   {
      return YetAnotherDuck(a) + YetAnotherDuck(b) + 1;
   }

   public static int YetAnotherDuck(int a)
   {
      if (a <= 0) return 0;
      return AnotherDuck(a - 1, a / 2);
   }

   public static int DocksAreNotDucks(int a, int b)
   {
      if (a == 0 || b == 0) return 0;
      return DocksAreNotDucks(a / b, b / a) * Duck(a, b);
   }

   public static void main(String[] args)
   {
      System.out.println(YetAnotherDuck(7));
   }
}
