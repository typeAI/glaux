package glaux.nn

case class Rectangle(x: Int, y: Int) {
  def *(factor: Double) = Rectangle((x * factor).toInt, (y * factor).toInt)
  def /(factor: Double) = *(1/factor)
  def +(toAdd: Int) = Rectangle(x + toAdd, y + toAdd)
  def -(toSubtract: Int) = +(- toSubtract)
  def +(p: Rectangle) = Rectangle(x + p.x, y + p.y)
}
