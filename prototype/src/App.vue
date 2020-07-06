<template>
  <v-app>
    <navigation></navigation>

    <v-content>
      <router-view></router-view>
    </v-content>
  </v-app>
</template>

<script>
import ActiveLearning from "./components/ActiveLearning";
import Navigation from "./components/Navigation";

export default {
  name: "app",
  data() {},
  methods: {},
  components: {
    activeLearning: ActiveLearning,
    navigation: Navigation
  }
};
</script>

<style scoped>
h1 {
  font-family: "Arial";
}
</style>
