package glaux.nn


case class Rectangle(x: Int, y: Int) {
  def *(factor: Double) = Rectangle((x * factor).toInt, (y * factor).toInt)
  def /(factor: Double) = *(1/factor)
  def +(toAdd: Int) = Rectangle(x + toAdd, y + toAdd)
  def -(toSubtract: Int): Rectangle = this.+(- toSubtract)
  def +(p: Rectangle) = Rectangle(x + p.x, y + p.y)
  def contains(tx: Int, ty: Int): Boolean = tx >= 0 && tx < x && ty >= 0 && ty < y
}

case class RectangleRange(x: Range, y: Range) {
  def contains(tx: Int, ty: Int): Boolean = x.contains(tx) && y.contains(ty)
}
