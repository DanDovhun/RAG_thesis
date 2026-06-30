class ReindexJourneys < ActiveRecord::Migration[7.0]
  disable_ddl_transaction!

  def up
    PgSearch::Multisearch.rebuild(Journey)
  end

  def down
  end
end
