package glaux.nn

trait Net {
  def forward(v: Vol, isTraining: Boolean = false)
}
